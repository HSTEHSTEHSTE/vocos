"""Hold out one deterministic enrollment utterance per LibriSpeech speaker.

The held-out enrollment file is removed from each Vocos target filelist. The
resulting combined enrollment list can be passed to
``extract_ecapa_speaker_embeddings.py`` without conditioning a target on its
own waveform.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-filelist", type=Path, required=True)
    parser.add_argument("--val-filelist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def speaker_id(audio_path: Path) -> str:
    value = audio_path.parent.parent.name
    if not value.isdigit():
        raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {audio_path}")
    return value


def read_filelist(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    paths = [Path(line) for line in path.read_text().splitlines()]
    if not paths:
        raise ValueError(f"Filelist is empty: {path}")
    for path_item in paths:
        if not path_item.is_file():
            raise FileNotFoundError(path_item)
    return paths


def write_filelist(path: Path, audio_paths: list[Path]) -> None:
    if not audio_paths:
        raise ValueError(f"Refusing to write an empty target filelist: {path}")
    path.write_text("\n".join(map(str, audio_paths)) + "\n")


def main() -> None:
    args = parse_args()
    train_paths = read_filelist(args.train_filelist)
    val_paths = read_filelist(args.val_filelist)
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for audio_path in train_paths + val_paths:
        by_speaker[speaker_id(audio_path)].append(audio_path)

    enrollments = {speaker: min(paths, key=str) for speaker, paths in by_speaker.items()}
    train_targets = [path for path in train_paths if path != enrollments[speaker_id(path)]]
    val_targets = [path for path in val_paths if path != enrollments[speaker_id(path)]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_target_path = args.output_dir / "librispeech_train_targets.txt"
    val_target_path = args.output_dir / "librispeech_dev-clean_targets.txt"
    enrollment_path = args.output_dir / "librispeech_enrollments.txt"
    write_filelist(train_target_path, train_targets)
    write_filelist(val_target_path, val_targets)
    enrollment_path.write_text("\n".join(map(str, sorted(enrollments.values(), key=str))) + "\n")
    (args.output_dir / "enrollment-selection.json").write_text(
        json.dumps(
            {
                "train_source": str(args.train_filelist),
                "val_source": str(args.val_filelist),
                "train_targets": len(train_targets),
                "val_targets": len(val_targets),
                "speaker_count": len(enrollments),
                "enrollment_by_speaker": {speaker: str(path) for speaker, path in sorted(enrollments.items())},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Selected {len(enrollments)} held-out enrollment utterances.")
    print(f"Training targets: {len(train_targets)} -> {train_target_path}")
    print(f"Validation targets: {len(val_targets)} -> {val_target_path}")
    print(f"Enrollments: {enrollment_path}")


if __name__ == "__main__":
    main()
