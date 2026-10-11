#!/usr/bin/env python3
"""Quickly total audio duration for one or more newline-delimited manifests.

The script reads audio container headers with SoundFile; it never decodes the
waveform.  Header reads are dispatched concurrently, which is substantially
faster than serial duration probes on large local corpora.

Example:
    python summarize_audio_manifest.py \
        --manifest train=/path/to/train.txt \
        --manifest validation=/path/to/val.txt \
        --workers 32 \
        --output duration_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass(frozen=True)
class Manifest:
    label: str
    path: Path


def parse_manifest(value: str) -> Manifest:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH") from error
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("Manifest label cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Manifest does not exist: {path}")
    return Manifest(label=label, path=path)


def read_paths(manifest: Path) -> list[Path]:
    paths: list[Path] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        path_text = line.split("\t", 1)[0].strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            raise ValueError(f"{manifest}:{line_number}: expected an absolute audio path")
        paths.append(path)
    if not paths:
        raise ValueError(f"Manifest is empty: {manifest}")
    return paths


def duration_seconds(path: Path) -> float:
    info = sf.info(path)
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate in {path}: {info.samplerate}")
    return float(info.frames) / float(info.samplerate)


def summarize(manifest: Manifest, workers: int) -> dict[str, object]:
    paths = read_paths(manifest.path)
    total_seconds = 0.0
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(duration_seconds, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                total_seconds += future.result()
            except Exception as error:  # Keep scanning so all broken inputs are reported together.
                failures.append({"path": str(path), "error": str(error)})

    if failures:
        preview = "; ".join(f"{item['path']}: {item['error']}" for item in failures[:3])
        raise RuntimeError(f"{manifest.label}: {len(failures)} audio-header reads failed ({preview})")

    return {
        "manifest": str(manifest.path),
        "num_files": len(paths),
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
    }


def write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=parse_manifest,
        metavar="LABEL=PATH",
        help="newline-delimited absolute-path manifest; repeat for multiple splits",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, max(1, os.cpu_count() or 1)),
        help="concurrent audio-header reads (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    labels = [manifest.label for manifest in args.manifest]
    if len(set(labels)) != len(labels):
        raise ValueError("Manifest labels must be unique")

    splits = {manifest.label: summarize(manifest, args.workers) for manifest in args.manifest}
    total_seconds = sum(float(split["total_seconds"]) for split in splits.values())
    report = {
        "splits": splits,
        "total_files": sum(int(split["num_files"]) for split in splits.values()),
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
        "workers": args.workers,
    }
    if args.output:
        write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
