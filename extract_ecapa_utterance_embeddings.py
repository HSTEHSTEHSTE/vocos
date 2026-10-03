"""Extract one frozen ECAPA-TDNN vector for every LibriSpeech utterance.

Embeddings are stored as ``<output-dir>/utterances/<LibriSpeech-relative-path>.npy``.
The extractor is resumable: existing vectors are skipped unless ``--overwrite``
is provided. It batches variable-duration waveforms on a GPU and recursively
splits an oversized batch if CUDA reports an out-of-memory error.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batch-samples",
        type=int,
        default=4_800_000,
        help="Maximum padded 16-kHz samples per GPU batch (default: 300 seconds).",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N utterances (for validation).")
    return parser.parse_args()


def output_path(audio_path: Path, audio_root: Path, utterances_dir: Path) -> Path:
    return (utterances_dir / audio_path.relative_to(audio_root)).with_suffix(".npy")


def load_mono_audio(audio_path: Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(audio_path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio.squeeze(0).contiguous()


def write_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, value)
    temporary_path.replace(path)


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def is_cuda_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def main() -> None:
    args = parse_args()
    if args.sample_rate != 16_000:
        raise ValueError("speechbrain/spkrec-ecapa-voxceleb expects 16-kHz audio; use --sample-rate 16000.")
    if args.batch_size < 1 or args.max_batch_samples < 1 or args.progress_every < 1:
        raise ValueError("batch-size, max-batch-samples, and progress-every must be positive.")
    if not args.audio_root.is_dir():
        raise NotADirectoryError(args.audio_root)

    audio_root = args.audio_root.resolve()
    output_dir = args.output_dir.resolve()
    utterances_dir = output_dir / "utterances"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "manifest.json"
    os.environ.setdefault("HF_HOME", str(output_dir / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(output_dir / ".cache" / "huggingface" / "hub"))

    audio_paths = sorted(audio_root.rglob("*.flac"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive when supplied.")
        audio_paths = audio_paths[:args.limit]
    if not audio_paths:
        raise ValueError(f"No FLAC files found below {audio_root}")

    pending_paths = [
        path
        for path in audio_paths
        if args.overwrite or not output_path(path, audio_root, utterances_dir).is_file()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        manifest_path,
        {
            "audio_root": str(audio_root),
            "model_source": args.model_source,
            "sample_rate": args.sample_rate,
            "embedding_dim": 192,
            "layout": "utterances/<LibriSpeech-relative-path>.npy",
            "total_utterances": len(audio_paths),
            "pending_at_start": len(pending_paths),
        },
    )
    print(f"Found {len(audio_paths)} utterances; {len(pending_paths)} require extraction.")

    from speechbrain.inference.speaker import EncoderClassifier

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    classifier = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(args.model_cache_dir),
        run_opts={"device": str(device)},
    )
    classifier.eval()

    counters = {"written": 0, "skipped": len(audio_paths) - len(pending_paths), "batches": 0}

    def update_progress() -> None:
        write_json(
            progress_path,
            {
                "audio_root": str(audio_root),
                "output_dir": str(output_dir),
                "total_utterances": len(audio_paths),
                "pending_at_start": len(pending_paths),
                **counters,
            },
        )

    def encode_and_write(records: list[tuple[Path, torch.Tensor]]) -> None:
        if not records:
            return
        lengths = torch.tensor([waveform.numel() for _, waveform in records], dtype=torch.float32)
        maximum_length = int(lengths.max().item())
        padded = torch.zeros((len(records), maximum_length), dtype=torch.float32)
        for index, (_, waveform) in enumerate(records):
            padded[index, : waveform.numel()] = waveform
        relative_lengths = (lengths / maximum_length).to(device)
        try:
            with torch.inference_mode():
                embeddings = classifier.encode_batch(padded.to(device), relative_lengths)
        except RuntimeError as error:
            if is_cuda_oom(error) and len(records) > 1:
                print(f"CUDA OOM for {len(records)} utterances; retrying as two smaller batches.")
                torch.cuda.empty_cache()
                midpoint = len(records) // 2
                encode_and_write(records[:midpoint])
                encode_and_write(records[midpoint:])
                return
            raise
        embeddings = embeddings.detach().float().cpu().reshape(len(records), -1)
        if embeddings.size(1) != 192:
            raise ValueError(f"Expected 192-D ECAPA embeddings, got shape {tuple(embeddings.shape)}")
        for (audio_path, _), embedding in zip(records, embeddings, strict=True):
            write_array(output_path(audio_path, audio_root, utterances_dir), embedding.numpy().astype(np.float32))
            counters["written"] += 1
        counters["batches"] += 1
        if counters["batches"] % args.progress_every == 0:
            update_progress()
            print(
                f"Progress: written={counters['written']} skipped={counters['skipped']} "
                f"of {len(audio_paths)} utterances."
            )

    batch: list[tuple[Path, torch.Tensor]] = []
    padded_samples = 0
    for audio_path in pending_paths:
        waveform = load_mono_audio(audio_path, args.sample_rate)
        proposed_padded_samples = max(padded_samples, waveform.numel()) * (len(batch) + 1)
        if batch and (len(batch) >= args.batch_size or proposed_padded_samples > args.max_batch_samples):
            encode_and_write(batch)
            batch = []
            padded_samples = 0
        batch.append((audio_path, waveform))
        padded_samples = max(padded_samples, waveform.numel())
    encode_and_write(batch)
    update_progress()
    print(f"Completed ECAPA extraction: {counters['written']} written, {counters['skipped']} already present.")


if __name__ == "__main__":
    main()
