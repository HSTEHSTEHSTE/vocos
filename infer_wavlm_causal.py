"""Synthesize exactly one LibriSpeech sample with a WavLM-conditioned Vocos checkpoint."""

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from vocos.experiment import VocosExp


DEFAULT_ARTIFACT_ROOT = Path("/home/jhu/xli257/scratch_nandrew9/xli257/ARTS/vocos")
DEFAULT_SOURCE_ROOT = Path("/weka/scratch/jhu/nandrew9/corpora/LibriSpeech")
DEFAULT_FEATURE_ROOT = Path(
    "/weka/scratch/jhu/nandrew9/xli257/features/wavlm_streaming_l6_step10_history100_lookahead0/LibriSpeech"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wavlm-causal.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to use; defaults to the newest .ckpt in --run-dir.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "runs" / "wavlm_causal")
    parser.add_argument("--filelist", type=Path, default=DEFAULT_ARTIFACT_ROOT / "filelists" / "librispeech_dev-clean.txt")
    parser.add_argument("--sample-index", type=int, default=0, help="Zero-based filelist index; exactly one entry is used.")
    parser.add_argument("--audio", type=Path, help="Override the one source audio path selected from --filelist.")
    parser.add_argument("--feature", type=Path, help="Override the feature tensor paired with --audio.")
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--feature-source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--speaker-transform-dir",
        type=Path,
        help="Optional LinearVC speaker-transform directory. Lifts 75-D content features to the model's 1024-D input.",
    )
    parser.add_argument(
        "--speaker-embedding-dir",
        type=Path,
        help=(
            "Optional ECAPA root. Supports speakers/<speaker-id>.npy enrollment embeddings or "
            "utterances/<LibriSpeech-relative-path>.npy vectors; utterance roots use a deterministic "
            "non-target vector from the same speaker."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and checkpoint loading without synthesis.")
    return parser.parse_args()


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def instantiate_model(config: dict, checkpoint_path: Path, device: torch.device) -> VocosExp:
    model_config = config["model"]
    model_args = dict(model_config["init_args"])
    for component in ("feature_extractor", "backbone", "head"):
        component_config = model_args[component]
        component_cls = import_class(component_config["class_path"])
        model_args[component] = component_cls(**component_config["init_args"])

    model_cls = import_class(model_config["class_path"])
    model = model_cls(**model_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def newest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.rglob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found below {run_dir}")
    return checkpoints[0]


def selected_audio(args: argparse.Namespace) -> Path:
    if args.audio is not None:
        return args.audio
    entries = args.filelist.read_text().splitlines()
    if not 0 <= args.sample_index < len(entries):
        raise IndexError(f"sample index {args.sample_index} is outside the {len(entries)}-entry filelist")
    return Path(entries[args.sample_index])


def paired_feature(args: argparse.Namespace, audio_path: Path) -> Path:
    if args.feature is not None:
        return args.feature
    try:
        relative_audio_path = audio_path.relative_to(args.feature_source_root)
    except ValueError as exc:
        raise ValueError("--audio is outside --feature-source-root; pass --feature explicitly.") from exc
    return (args.feature_root / relative_audio_path).with_suffix(".pt")


def load_feature_tensor(feature_path: Path) -> torch.Tensor:
    try:
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
    except TypeError:
        features = torch.load(feature_path, map_location="cpu")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"Expected a rank-2 feature tensor in {feature_path}, got {type(features)!r}.")
    return features.float()


def lift_features_for_librispeech_speaker(features: torch.Tensor, audio_path: Path, transform_dir: Path) -> torch.Tensor:
    speaker_id = audio_path.parent.parent.name
    if not speaker_id.isdigit():
        raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {audio_path}")
    speakers_dir = transform_dir / "speakers"
    transform_path = (speakers_dir if speakers_dir.is_dir() else transform_dir) / f"{speaker_id}.npy"
    if not transform_path.is_file():
        raise FileNotFoundError(f"Speaker transform is missing: {transform_path}")
    transform = torch.from_numpy(np.load(transform_path)).float().contiguous()
    if transform.ndim != 2 or transform.size(0) != features.size(1):
        raise ValueError(
            f"Transform {transform_path} must have shape ({features.size(1)}, output_dim), got {tuple(transform.shape)}."
        )
    return features @ transform


def load_speaker_embedding_for_librispeech_speaker(
    audio_path: Path, embedding_dir: Path, feature_source_root: Path
) -> tuple[torch.Tensor, Path]:
    speaker_id = audio_path.parent.parent.name
    if not speaker_id.isdigit():
        raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {audio_path}")
    speakers_dir = embedding_dir / "speakers"
    if speakers_dir.is_dir():
        embedding_path = speakers_dir / f"{speaker_id}.npy"
    else:
        utterances_dir = embedding_dir / "utterances"
        utterances_dir = utterances_dir if utterances_dir.is_dir() else embedding_dir
        try:
            relative_audio_path = audio_path.relative_to(feature_source_root)
        except ValueError as exc:
            raise ValueError("Audio is outside --feature-source-root; cannot find its utterance ECAPA vector.") from exc
        if len(relative_audio_path.parts) < 4:
            raise ValueError(f"Expected a LibriSpeech-relative audio path, got {relative_audio_path}")
        target_embedding_path = (utterances_dir / relative_audio_path).with_suffix(".npy")
        speaker_dir = utterances_dir / relative_audio_path.parts[0] / speaker_id
        candidates = sorted(path for path in speaker_dir.rglob("*.npy") if path != target_embedding_path)
        if not candidates:
            raise FileNotFoundError(
                f"No non-target utterance ECAPA vector found for LibriSpeech speaker {speaker_id} in {speaker_dir}"
            )
        embedding_path = candidates[0]
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Speaker embedding is missing: {embedding_path}")
    embedding = torch.from_numpy(np.load(embedding_path)).float().contiguous().squeeze()
    if embedding.ndim != 1:
        raise ValueError(f"Expected a rank-1 speaker embedding in {embedding_path}, got {tuple(embedding.shape)}.")
    return embedding, embedding_path


def align_features(features: torch.Tensor, target_frames: int) -> torch.Tensor:
    if features.size(0) == 0:
        raise ValueError("Feature tensor has no frames.")
    if features.size(0) < target_frames:
        features = torch.cat((features, features[-1:].expand(target_frames - features.size(0), -1)), dim=0)
    return features[:target_frames]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    checkpoint_path = args.checkpoint or newest_checkpoint(args.run_dir)
    audio_path = selected_audio(args)
    feature_path = paired_feature(args, audio_path)
    for path in (args.config, checkpoint_path, audio_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.config.read_text())
    data_config = config["data"]["init_args"]["val_params"]
    sample_rate = int(data_config["sampling_rate"])
    feature_hop_length = int(data_config["feature_hop_length"])

    source_audio, source_rate = torchaudio.load(audio_path)
    if source_audio.size(0) > 1:
        source_audio = source_audio.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        source_audio = torchaudio.functional.resample(source_audio, source_rate, sample_rate)

    features = load_feature_tensor(feature_path)
    if args.speaker_transform_dir is not None:
        features = lift_features_for_librispeech_speaker(features, audio_path, args.speaker_transform_dir)
    speaker_embedding = None
    speaker_embedding_path = None
    if args.speaker_embedding_dir is not None:
        speaker_embedding, speaker_embedding_path = load_speaker_embedding_for_librispeech_speaker(
            audio_path, args.speaker_embedding_dir, args.feature_source_root
        )
    target_frames = math.ceil(source_audio.size(-1) / feature_hop_length)
    features = align_features(features, target_frames)
    model = instantiate_model(config, checkpoint_path, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Audio: {audio_path}")
    print(f"Features: {feature_path}")
    print(f"Feature shape: {tuple(features.shape)}")
    if args.dry_run:
        print("Dry run succeeded.")
        return

    with torch.inference_mode():
        prediction = model(
            source_audio.to(device),
            precomputed_features=features.transpose(0, 1).unsqueeze(0).to(device),
            speaker_embedding=None if speaker_embedding is None else speaker_embedding.unsqueeze(0).to(device),
        )[0].float().cpu().clamp(-1, 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_audio = args.output_dir / f"{audio_path.stem}_pred.wav"
    output_metadata = args.output_dir / f"{audio_path.stem}_pred.json"
    torchaudio.save(output_audio, prediction.unsqueeze(0), sample_rate)
    output_metadata.write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "feature": str(feature_path),
                "checkpoint": str(checkpoint_path),
                "speaker_embedding_dir": None if args.speaker_embedding_dir is None else str(args.speaker_embedding_dir),
                "speaker_embedding": None if speaker_embedding_path is None else str(speaker_embedding_path),
                "prediction": str(output_audio),
                "sample_rate": sample_rate,
                "input_samples": source_audio.size(-1),
                "output_samples": prediction.numel(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote: {output_audio}")
    print(f"Metadata: {output_metadata}")


if __name__ == "__main__":
    main()
