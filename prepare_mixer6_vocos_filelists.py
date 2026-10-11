#!/usr/bin/env python3
"""Create deterministic speaker-aware Mixer6 train/validation filelists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


SPEAKER_RE = re.compile(r"_(?:LDC|HRM)_(\d+)_CH\d+\.flac$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixer6-root", type=Path, required=True)
    parser.add_argument("--speaker-transform-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.validation_percent < 50:
        parser.error("--validation-percent must be in [1, 49]")
    return args


def speaker_id(path: Path) -> str | None:
    match = SPEAKER_RE.search(path.name)
    return None if match is None else match.group(1)


def main() -> None:
    args = parse_args()
    audio_root = args.mixer6_root / "data" / "pcm_flac"
    roster_path = args.mixer6_root / "docs" / "mx6_subjs.csv"
    transforms_root = args.speaker_transform_dir / "speakers"
    if not transforms_root.is_dir():
        transforms_root = args.speaker_transform_dir
    if not audio_root.is_dir() or not roster_path.is_file() or not transforms_root.is_dir():
        raise FileNotFoundError("Mixer6 audio, roster, or packed speaker-transform directory is missing")
    with roster_path.open(newline="") as handle:
        roster = {row["subjid"].strip() for row in csv.DictReader(handle) if row.get("subjid", "").strip()}
    available = {path.stem for path in transforms_root.glob("*.npy")}
    eligible = roster & available
    train: list[str] = []
    validation: list[str] = []
    skipped = 0
    for path in sorted(audio_root.rglob("*.flac")):
        speaker = speaker_id(path)
        if speaker not in eligible:
            skipped += 1
            continue
        bucket = int.from_bytes(hashlib.blake2b(str(path).encode(), digest_size=8).digest(), "little") % 100
        (validation if bucket < args.validation_percent else train).append(str(path))
    if not train or not validation:
        raise RuntimeError(f"Empty split: train={len(train)}, validation={len(validation)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(train) + "\n")
    (args.output_dir / "val.txt").write_text("\n".join(validation) + "\n")
    (args.output_dir / "summary.txt").write_text(
        f"eligible_speakers={len(eligible)}\ntrain_files={len(train)}\nval_files={len(validation)}\nskipped_files={skipped}\n"
    )
    print((args.output_dir / "summary.txt").read_text(), end="")


if __name__ == "__main__":
    main()
