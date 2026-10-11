"""Synthesize one Mixer6 segment with the distilled one-frame WavLM Vocos model."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from vocos.experiment import VocosExp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--speaker-matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Segment length after --start-seconds. Mutually exclusive with --end-seconds.",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--end-seconds",
        type=float,
        help="Exclusive source-audio endpoint. Use this for a full, timestamped utterance.",
    )
    parser.add_argument(
        "--full-audio",
        action="store_true",
        help="Synthesize the complete source utterance after resampling.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def instantiate_model(config: dict, checkpoint_path: Path, device: torch.device) -> VocosExp:
    model_args = dict(config["model"]["init_args"])
    for component in ("feature_extractor", "backbone", "head"):
        component_config = model_args[component]
        model_args[component] = import_class(component_config["class_path"])(**component_config["init_args"])
    model = import_class(config["model"]["class_path"])(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.start_seconds < 0:
        raise ValueError("--start-seconds must be non-negative")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    selection_count = sum(value is not None for value in (args.num_samples, args.end_seconds)) + int(args.full_audio)
    if selection_count != 1:
        raise ValueError("Use exactly one of --num-samples, --end-seconds, or --full-audio")
    if args.end_seconds is not None and args.end_seconds <= args.start_seconds:
        raise ValueError("--end-seconds must be greater than --start-seconds")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for path in (args.config, args.checkpoint, args.audio):
        if not path.is_file():
            raise FileNotFoundError(path)

    matrix_root = args.speaker_matrix_dir / "speakers"
    matrix_path = (matrix_root if matrix_root.is_dir() else args.speaker_matrix_dir) / f"{args.speaker_id}.npy"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    speaker_matrix = torch.from_numpy(np.load(matrix_path)).float().contiguous()
    if tuple(speaker_matrix.shape) != (75, 1024):
        raise ValueError(f"Expected a (75, 1024) matrix, got {tuple(speaker_matrix.shape)} from {matrix_path}")

    config = yaml.safe_load(args.config.read_text())
    sample_rate = int(config["data"]["init_args"]["val_params"]["sampling_rate"])
    audio, source_rate = torchaudio.load(args.audio)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Mirror the current Mixer6 training data path exactly: normalize each
    # complete recording to -3 dBFS, then resample to 24 kHz. The explicit
    # time range is applied afterward, so this is a real utterance-sized
    # segment rather than a fixed crop from the recording start.
    target_peak = 10 ** (-3 / 20)
    peak = audio.abs().amax()
    if peak > 0:
        audio = audio * (target_peak / peak)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    start_sample = round(args.start_seconds * sample_rate)
    if args.full_audio:
        end_sample = audio.size(-1)
    elif args.end_seconds is not None:
        end_sample = round(args.end_seconds * sample_rate)
    else:
        end_sample = start_sample + args.num_samples
    if start_sample >= audio.size(-1):
        raise ValueError(f"--start-seconds={args.start_seconds} starts after the recording ends")
    audio = audio[:, start_sample : min(end_sample, audio.size(-1))]
    if audio.size(-1) == 0:
        raise ValueError("Selected segment is empty")
    num_samples = audio.size(-1)

    model = instantiate_model(config, args.checkpoint, device)
    with torch.inference_mode():
        prediction = model(
            audio.to(device),
            speaker_matrix=speaker_matrix.unsqueeze(0).to(device),
        )[0].float().cpu().clamp(-1, 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_end = args.start_seconds + num_samples / sample_rate
    segment_tag = f"{args.start_seconds:.3f}_{segment_end:.3f}"
    output_audio = args.output_dir / f"{args.audio.stem}_{segment_tag}_pred.wav"
    reference_audio = args.output_dir / f"{args.audio.stem}_{segment_tag}_reference.wav"
    output_metadata = output_audio.with_suffix(".json")
    torchaudio.save(output_audio, prediction.unsqueeze(0), sample_rate)
    torchaudio.save(reference_audio, audio, sample_rate)
    output_metadata.write_text(
        json.dumps(
            {
                "audio": str(args.audio),
                "checkpoint": str(args.checkpoint),
                "speaker_id": args.speaker_id,
                "speaker_matrix": str(matrix_path),
                "sample_rate": sample_rate,
                "start_seconds": args.start_seconds,
                "end_seconds": segment_end,
                "num_samples": num_samples,
                "reference": str(reference_audio),
                "prediction": str(output_audio),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote: {output_audio}")
    print(f"Reference: {reference_audio}")
    print(f"Metadata: {output_metadata}")


if __name__ == "__main__":
    main()
