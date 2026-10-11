"""Synthesize a 16 kHz Vocos checkpoint from precomputed 75-D content features.

This entry point intentionally has no WavLM dependency.  It decodes a complete
utterance from a ``[frames, 75]`` torch tensor and one target speaker transform
with shape ``[75, 1024]``.  The checkpoint YAML defines the sample rate and
hop length, so it supports the 16 kHz / 320-sample-hop lifted-content model
without inheriting assumptions from the older 24 kHz inference scripts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from infer_wavlm_causal import decode_precomputed_features_streaming, instantiate_model, load_feature_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Checkpoint training YAML.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning checkpoint to decode.")
    parser.add_argument("--feature", type=Path, required=True, help="Torch tensor with shape [frames, 75].")
    parser.add_argument(
        "--speaker-transform",
        type=Path,
        required=True,
        help="NumPy target-speaker transform with shape [75, 1024].",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output 16 kHz waveform path.")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cpu.")
    parser.add_argument(
        "--stream-chunk-frames",
        type=int,
        default=10,
        help="Cached Vocos chunk size in chronological feature frames (default: 10).",
    )
    args = parser.parse_args()
    if args.stream_chunk_frames <= 0:
        parser.error("--stream-chunk-frames must be positive.")
    return args


def model_geometry(config: dict) -> tuple[int, int, int, int]:
    """Return sample rate, synthesis hop, content dimension, and model input dimension."""

    model_args = config["model"]["init_args"]
    data_args = config["data"]["init_args"]["val_params"]
    sample_rate = int(model_args["sample_rate"])
    hop_length = int(model_args["head"]["init_args"]["hop_length"])
    content_dim = int(data_args["feature_dim"])
    model_dim = int(model_args["feature_extractor"]["init_args"]["feature_dim"])
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError(f"Invalid model geometry: sample_rate={sample_rate}, hop_length={hop_length}.")
    return sample_rate, hop_length, content_dim, model_dim


def load_speaker_transform(path: Path, content_dim: int, model_dim: int) -> torch.Tensor:
    transform = torch.from_numpy(np.asarray(np.load(path), dtype=np.float32)).contiguous()
    expected = (content_dim, model_dim)
    if tuple(transform.shape) != expected:
        raise ValueError(f"Expected speaker transform {path} to have shape {expected}, got {tuple(transform.shape)}.")
    return transform


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu instead.")
    for path in (args.config, args.checkpoint, args.feature, args.speaker_transform):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.config.read_text())
    sample_rate, hop_length, content_dim, model_dim = model_geometry(config)
    content = load_feature_tensor(args.feature)
    if content.size(0) == 0 or content.size(1) != content_dim:
        raise ValueError(
            f"Expected non-empty content tensor [frames, {content_dim}] in {args.feature}, got {tuple(content.shape)}."
        )
    speaker_transform = load_speaker_transform(args.speaker_transform, content_dim, model_dim)
    lifted = content @ speaker_transform

    model = instantiate_model(config, args.checkpoint, device)
    with torch.inference_mode():
        model_features = lifted.transpose(0, 1).unsqueeze(0).to(device)
        prediction = decode_precomputed_features_streaming(model, model_features, args.stream_chunk_frames)
        prediction = prediction[0].float().cpu().clamp(-1, 1)

    expected_samples = content.size(0) * hop_length
    if prediction.numel() < expected_samples:
        raise RuntimeError(
            f"Decoder returned {prediction.numel()} samples for {content.size(0)} frames; expected at least {expected_samples}."
        )
    prediction = prediction[:expected_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, prediction.unsqueeze(0), sample_rate)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "content_feature": str(args.feature),
                "speaker_transform": str(args.speaker_transform),
                "content_frames": content.size(0),
                "content_dim": content_dim,
                "model_input_dim": model_dim,
                "sample_rate": sample_rate,
                "hop_length": hop_length,
                "output_samples": prediction.numel(),
                "inference_mode": "cached_streaming_vocos",
                "stream_chunk_frames": args.stream_chunk_frames,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote: {args.output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
