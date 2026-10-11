#!/usr/bin/env python3
"""Extract original WavLM-Large layer-6 features on CPU for a small manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True, help="TSV: unique tag then audio path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavlm-source", type=Path, required=True)
    parser.add_argument("--wavlm-checkpoint", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    return args


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            tag, audio_text = line.split("\t", maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Expected tag and audio path at {path}:{number}") from error
        audio = Path(audio_text)
        if not tag or tag in seen or not audio.is_file():
            raise ValueError(f"Invalid or duplicate manifest entry at {path}:{number}: {line!r}")
        rows.append((tag, audio))
        seen.add(tag)
    if not rows:
        raise ValueError(f"No inputs in {path}")
    return rows


def load_audio(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio: {path}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform.to(dtype=torch.float32).contiguous()


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if str(args.wavlm_source) not in sys.path:
        sys.path.insert(0, str(args.wavlm_source))
    from official_wavlm import load_official_wavlm_large  # pylint: disable=import-outside-toplevel

    entries = read_manifest(args.input_manifest)
    model, _ = load_official_wavlm_large(args.wavlm_checkpoint, device=torch.device("cpu"))
    model.eval()
    output_manifest: list[str] = []
    for tag, audio_path in entries:
        destination = args.output_dir / f"{tag}.pt"
        if not (args.skip_existing and destination.is_file()):
            waveform = load_audio(audio_path)
            with torch.inference_mode():
                features, _ = model.extract_features(waveform, output_layer=6)
            if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != 1024:
                raise RuntimeError(f"Unexpected layer-6 feature shape for {audio_path}: {tuple(features.shape)}")
            save_tensor(features[0].cpu().to(dtype=torch.float16).contiguous(), destination)
        output_manifest.append(f"{tag}\t{audio_path}\t{destination}\n")
        print(f"Prepared {tag}: {destination}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "features.tsv").write_text("".join(output_manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
