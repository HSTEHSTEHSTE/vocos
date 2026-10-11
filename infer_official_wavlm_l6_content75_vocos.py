#!/usr/bin/env python3
"""Synthesize full utterances through official chunked WavLM-L6 and cached Vocos.

The feature path is intentionally explicit and matches the requested deployment
geometry: original Microsoft WavLM-Large layer 6 is re-evaluated on 200 ms
cores with bounded left waveform history and the 80-sample frontend tail.  Each
emitted L6 block is projected through UTXSS, raised with the source
LibriSpeech speaker's 75x1024 transform, and decoded immediately through the
stateful causal Vocos decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from infer_wavlm_causal import instantiate_model


WAVLM_SAMPLE_RATE = 16_000
WAVLM_HOP_SAMPLES = 320
WAVLM_RECEPTIVE_FIELD = 400
WAVLM_FRONTEND_TAIL = WAVLM_RECEPTIVE_FIELD - WAVLM_HOP_SAMPLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, required=True, help="One full source-audio path per line.")
    parser.add_argument("--num-utterances", type=int, default=10)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wavlm-source", type=Path, required=True)
    parser.add_argument("--wavlm-checkpoint", type=Path, required=True)
    parser.add_argument("--utxss", type=Path, required=True)
    parser.add_argument("--speaker-transform-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--core-ms", type=float, default=200.0)
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument(
        "--vocos-decode-mode",
        choices=("cached", "independent-chunks", "one-pass"),
        default="cached",
        help="Use cached causal chunks, independent chunk forwards, or one full Vocos forward pass after chunked WavLM extraction.",
    )
    parser.add_argument(
        "--verify-vocos-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally record cached Vocos divergence from a one-pass decode; never gates synthesis.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.num_utterances <= 0:
        parser.error("--num-utterances must be positive")
    if args.core_ms <= 0 or args.history_seconds < 0:
        parser.error("--core-ms must be positive and --history-seconds must be non-negative")
    args.core_samples = round(args.core_ms * WAVLM_SAMPLE_RATE / 1000)
    args.history_samples = round(args.history_seconds * WAVLM_SAMPLE_RATE)
    if args.core_samples <= 0 or args.core_samples % WAVLM_HOP_SAMPLES:
        parser.error("--core-ms must correspond to an integer multiple of the 20 ms WavLM hop")
    if args.history_samples % WAVLM_HOP_SAMPLES:
        parser.error("--history-seconds must correspond to an integer multiple of the 20 ms WavLM hop")
    return args


def read_sources(filelist: Path, count: int) -> list[Path]:
    if not filelist.is_file():
        raise FileNotFoundError(filelist)
    paths = [Path(line) for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) < count:
        raise ValueError(f"{filelist} has only {len(paths)} entries; need {count}")
    selected = paths[:count]
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source audio is missing: {missing[0]}")
    if len({path.stem for path in selected}) != len(selected):
        raise ValueError("Selected utterance basenames must be unique")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_audio(path: Path, target_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform.float().contiguous()


def speaker_id_for_librispeech(path: Path) -> str:
    speaker_id = path.parent.parent.name
    if not speaker_id.isdigit():
        raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {path}")
    return speaker_id


def speaker_transform_path(root: Path, speaker_id: str) -> Path:
    speakers = root / "speakers"
    path = (speakers if speakers.is_dir() else root) / f"{speaker_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Speaker transform is missing: {path}")
    return path


def wavlm_chunk_bounds(
    waveform_length: int, current_start: int, core_samples: int, history_samples: int
) -> tuple[int, int, int]:
    """Return one training-aligned WavLM overlap chunk and its emitted core end."""
    core_end = min(current_start + core_samples, waveform_length)
    chunk_start = max(current_start - history_samples, 0)
    chunk_end = min(core_end + WAVLM_FRONTEND_TAIL, waveform_length)
    return chunk_start, chunk_end, core_end


def extract_l6_blocks(
    wavlm: torch.nn.Module,
    waveform: torch.Tensor,
    core_samples: int,
    history_samples: int,
) -> list[torch.Tensor]:
    """Yield only chronological new L6 frames from overlap-recomputed WavLM."""
    blocks: list[torch.Tensor] = []
    current_start = 0
    waveform_length = waveform.size(-1)
    while current_start + WAVLM_RECEPTIVE_FIELD <= waveform_length:
        chunk_start, chunk_end, core_end = wavlm_chunk_bounds(
            waveform_length, current_start, core_samples, history_samples
        )
        features, _ = wavlm.extract_features(waveform[:, chunk_start:chunk_end], output_layer=6)
        emit_start = (current_start - chunk_start) // WAVLM_HOP_SAMPLES
        if core_end >= waveform_length:
            emitted = features[:, emit_start:]
        else:
            emitted = features[:, emit_start : emit_start + core_samples // WAVLM_HOP_SAMPLES]
        if emitted.size(1) == 0:
            raise RuntimeError(f"WavLM emitted no frames for core beginning at sample {current_start}")
        blocks.append(emitted)
        if core_end >= waveform_length:
            break
        current_start += core_samples
    if not blocks:
        raise RuntimeError("Input is shorter than WavLM's 25 ms receptive field")
    return blocks


def align_feature_blocks(blocks: list[torch.Tensor], expected_frames: int) -> list[torch.Tensor]:
    observed_frames = sum(block.size(1) for block in blocks)
    if observed_frames > expected_frames:
        raise RuntimeError(f"WavLM emitted {observed_frames} frames; expected at most {expected_frames}")
    if observed_frames < expected_frames:
        # The valid WavLM convolution can stop up to one partial 200 ms core
        # early at an utterance boundary. Mirror the Vocos training/inference
        # feature aligner by holding the final available L6 vector.
        blocks.append(blocks[-1][:, -1:].expand(-1, expected_frames - observed_frames, -1).clone())
    return blocks


def atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.config, args.checkpoint, args.wavlm_source, args.wavlm_checkpoint, args.utxss):
        if not path.exists():
            raise FileNotFoundError(path)
    sources = read_sources(args.filelist, args.num_utterances)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_args = config["model"]["init_args"]
    sample_rate = int(model_args["sample_rate"])
    head_args = model_args["head"]["init_args"]
    vocos_hop = int(head_args["hop_length"])
    if sample_rate != 24_000 or vocos_hop != 480:
        raise ValueError(f"This evaluator expects the 24 kHz / 480-sample Vocos geometry, got {sample_rate}/{vocos_hop}")

    if str(args.wavlm_source) not in sys.path:
        sys.path.insert(0, str(args.wavlm_source))
    wavlm_loader = importlib.import_module("official_wavlm")
    wavlm, _ = wavlm_loader.load_official_wavlm_large(args.wavlm_checkpoint, device=device)
    wavlm.requires_grad_(False).eval()
    vocos = instantiate_model(config, args.checkpoint, device)
    utxss = torch.from_numpy(np.load(args.utxss)).to(device=device, dtype=torch.float32).contiguous()
    if tuple(utxss.shape) != (1024, 75):
        raise ValueError(f"Expected UTXSS shape (1024, 75), got {tuple(utxss.shape)}")

    records: list[dict[str, object]] = []
    transform_cache: dict[str, tuple[torch.Tensor, Path]] = {}
    for index, source_path in enumerate(sources, start=1):
        output_path = args.output_dir / "wavs" / f"{source_path.stem}_official_l6_hist5s_content75_lift_vocos.wav"
        metadata_path = output_path.with_suffix(".json")
        if args.skip_existing and output_path.is_file() and metadata_path.is_file():
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            records.append(record)
            print(f"[{index}/{len(sources)}] Reusing {output_path.name}", flush=True)
            continue

        started = time.perf_counter()
        wav16 = load_audio(source_path, WAVLM_SAMPLE_RATE).to(device)
        speaker_id = speaker_id_for_librispeech(source_path)
        if speaker_id not in transform_cache:
            transform_path = speaker_transform_path(args.speaker_transform_dir, speaker_id)
            transform = torch.from_numpy(np.load(transform_path)).to(device=device, dtype=torch.float32).contiguous()
            if tuple(transform.shape) != (75, 1024):
                raise ValueError(f"Expected 75x1024 speaker transform at {transform_path}, got {tuple(transform.shape)}")
            transform_cache[speaker_id] = transform, transform_path
        transform, transform_path = transform_cache[speaker_id]

        with torch.inference_mode():
            l6_blocks = extract_l6_blocks(wavlm, wav16, args.core_samples, args.history_samples)
            expected_frames = math.ceil(wav16.size(-1) / WAVLM_HOP_SAMPLES)
            l6_blocks = align_feature_blocks(l6_blocks, expected_frames)
            vocos_state = None
            audio_blocks: list[torch.Tensor] = []
            raised_blocks: list[torch.Tensor] = []
            content_blocks: list[torch.Tensor] = []
            for l6_block in l6_blocks:
                content = l6_block.float() @ utxss
                raised = content @ transform
                content_blocks.append(content.squeeze(0).cpu())
                raised_blocks.append(raised)
                if args.vocos_decode_mode == "cached":
                    audio_block, vocos_state = vocos.decode_stream(raised.transpose(1, 2), vocos_state)
                    audio_blocks.append(audio_block)
                elif args.vocos_decode_mode == "independent-chunks":
                    audio_blocks.append(vocos(torch.empty(1, 1, device=device), precomputed_features=raised.transpose(1, 2)))
            full_features = torch.cat(raised_blocks, dim=1).transpose(1, 2)
            if args.vocos_decode_mode in {"cached", "independent-chunks"}:
                prediction = torch.cat(audio_blocks, dim=-1)
            else:
                prediction = vocos(torch.empty(1, 1, device=device), precomputed_features=full_features)
            content75 = torch.cat(content_blocks, dim=0)
            verification: dict[str, object] | None = None
            if args.verify_vocos_streaming and args.vocos_decode_mode == "cached":
                one_pass = vocos(torch.empty(1, 1, device=device), precomputed_features=full_features)
                max_abs_error = float((prediction - one_pass).abs().max().item())
                verification = {
                    "max_abs_error": max_abs_error,
                    "enforced": False,
                }
            prediction = prediction[0].float().cpu().clamp(-1, 1)

        feature_path = args.output_dir / "content75" / f"{source_path.stem}.pt"
        atomic_torch_save(content75, feature_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, prediction.unsqueeze(0), sample_rate)
        record = {
            "source_audio": str(source_path),
            "speaker_id": speaker_id,
            "speaker_transform": str(transform_path),
            "output_wav": str(output_path),
            "content75": str(feature_path),
            "source_samples_16khz": int(wav16.size(-1)),
            "output_samples_24khz": int(prediction.numel()),
            "feature_frames": int(content75.size(0)),
            "vocos_stream_chunk_frames": args.core_samples // WAVLM_HOP_SAMPLES,
            "vocos_decode_mode": args.vocos_decode_mode,
            "vocos_streaming_verification": verification,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json_save(record, metadata_path)
        records.append(record)
        print(
            f"[{index}/{len(sources)}] {source_path.name}: frames={content75.size(0)} "
            f"seconds={record['elapsed_seconds']:.2f} -> {output_path}",
            flush=True,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config),
        "wavlm": {
            "source": str(args.wavlm_source),
            "checkpoint": str(args.wavlm_checkpoint),
            "layer": 6,
            "core_ms": args.core_ms,
            "core_samples": args.core_samples,
            "history_seconds": args.history_seconds,
            "history_samples": args.history_samples,
            "lookahead_samples": 0,
            "frontend_tail_samples": WAVLM_FRONTEND_TAIL,
            "mode": "overlap-recomputed original WavLM-Large",
        },
        "content_projection": {"path": str(args.utxss), "sha256": sha256(args.utxss), "shape": [1024, 75]},
        "speaker_transform_dir": str(args.speaker_transform_dir),
        "vocos": {"sample_rate": sample_rate, "hop_samples": vocos_hop, "decode_mode": args.vocos_decode_mode},
        "records": records,
    }
    atomic_json_save(summary, args.output_dir / "manifest.json")
    print(f"Wrote {len(records)} full-utterance WAVs under {args.output_dir / 'wavs'}", flush=True)


if __name__ == "__main__":
    main()
