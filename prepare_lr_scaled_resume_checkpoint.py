#!/usr/bin/env python3
"""Create a safe learning-rate-scaled copy of a Lightning training checkpoint.

The copy retains model weights, optimizer moments, scheduler position, and
training progress.  Optimizer and cosine-scheduler learning-rate values are
scaled together, which makes a true continuation possible without mutating the
source checkpoint.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def positive_scale(value: str) -> float:
    scale = float(value)
    if not 0.0 < scale <= 1.0:
        raise argparse.ArgumentTypeError("scale must be in (0, 1]")
    return scale


def scale_lr_values(state: dict[str, Any], scale: float) -> list[float]:
    values: list[float] = []
    for group in state.get("param_groups", []):
        if "lr" not in group:
            raise ValueError("optimizer param group has no learning rate")
        old_lr = float(group["lr"])
        group["lr"] = old_lr * scale
        if "initial_lr" in group:
            group["initial_lr"] = float(group["initial_lr"]) * scale
        values.append(old_lr)
    return values


def scale_scheduler_values(state: dict[str, Any], scale: float) -> None:
    for key in ("base_lrs", "_last_lr"):
        if key in state:
            state[key] = [float(value) * scale for value in state[key]]


def checkpoint_is_expected(checkpoint: dict[str, Any], source: Path, scale: float) -> bool:
    metadata = checkpoint.get("vocos_lr_scaled_resume")
    return isinstance(metadata, dict) and metadata.get("source_checkpoint") == str(source) and metadata.get("scale") == scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=positive_scale)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise SystemExit("output must differ from source; refusing to mutate the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        if checkpoint_is_expected(existing, source, args.scale):
            print(f"Reusing prepared checkpoint: {output}")
            return
        raise SystemExit(f"Refusing to overwrite a different checkpoint: {output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    optimizer_states = checkpoint.get("optimizer_states")
    scheduler_states = checkpoint.get("lr_schedulers")
    if not optimizer_states or not scheduler_states:
        raise SystemExit("source checkpoint has no optimizer and scheduler states to resume")
    if len(optimizer_states) != len(scheduler_states):
        raise SystemExit("optimizer and scheduler state counts do not match")

    old_lrs = [scale_lr_values(state, args.scale) for state in optimizer_states]
    for state in scheduler_states:
        scale_scheduler_values(state, args.scale)

    checkpoint["vocos_lr_scaled_resume"] = {
        "source_checkpoint": str(source),
        "scale": args.scale,
        "old_optimizer_lrs": old_lrs,
        "new_optimizer_lrs": [[value * args.scale for value in values] for values in old_lrs],
    }

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote learning-rate-scaled resume checkpoint: {output}")
    print(f"Scale: {args.scale}")
    print(f"Old optimizer LRs: {old_lrs}")


if __name__ == "__main__":
    main()
