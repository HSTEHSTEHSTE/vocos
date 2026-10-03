"""Create frozen ECAPA-TDNN enrollment embeddings for LibriSpeech speakers.

The enrollment filelist must contain only reference utterances that are separate
from the target utterances used by Vocos training or validation. All enrollment
utterances for a speaker are L2-normalized, averaged, and normalized again.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrollment-filelist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def speaker_id_from_librispeech_path(audio_path: Path) -> str:
    speaker_id = audio_path.parent.parent.name
    if not speaker_id.isdigit():
        raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {audio_path}")
    return speaker_id


def load_mono_audio(audio_path: Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(audio_path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio.squeeze(0)


def main() -> None:
    args = parse_args()
    if not args.enrollment_filelist.is_file():
        raise FileNotFoundError(args.enrollment_filelist)
    if args.sample_rate != 16_000:
        raise ValueError("speechbrain/spkrec-ecapa-voxceleb expects 16 kHz audio; use --sample-rate 16000.")

    output_dir = args.output_dir
    speakers_dir = output_dir / "speakers"
    model_cache_dir = args.model_cache_dir or output_dir / ".cache" / "speechbrain"
    os.environ.setdefault("HF_HOME", str(output_dir / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(output_dir / ".cache" / "huggingface" / "hub"))

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for line in args.enrollment_filelist.read_text().splitlines():
        audio_path = Path(line)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        grouped_paths[speaker_id_from_librispeech_path(audio_path)].append(audio_path)
    if not grouped_paths:
        raise ValueError("Enrollment filelist is empty.")

    from speechbrain.inference.speaker import EncoderClassifier

    device = torch.device(args.device)
    classifier = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(model_cache_dir),
        run_opts={"device": str(device)},
    )
    speakers_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_source": args.model_source,
        "sample_rate": args.sample_rate,
        "embedding_dim": 192,
        "enrollment_filelist": str(args.enrollment_filelist),
        "speakers": {},
    }

    for speaker_id, audio_paths in sorted(grouped_paths.items()):
        output_path = speakers_dir / f"{speaker_id}.npy"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace it.")
        embeddings = []
        for audio_path in audio_paths:
            audio = load_mono_audio(audio_path, args.sample_rate).unsqueeze(0).to(device)
            with torch.inference_mode():
                embedding = classifier.encode_batch(audio).squeeze().float()
            embeddings.append(torch.nn.functional.normalize(embedding, dim=0))
        mean_embedding = torch.nn.functional.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
        if mean_embedding.numel() != 192:
            raise ValueError(f"Expected a 192-D ECAPA embedding, got {mean_embedding.numel()} for speaker {speaker_id}.")
        np.save(output_path, mean_embedding.cpu().numpy().astype(np.float32))
        manifest["speakers"][speaker_id] = {"enrollment_paths": [str(path) for path in audio_paths]}
        print(f"Wrote {output_path} from {len(audio_paths)} enrollment utterance(s).")

    (output_dir / "enrollment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
