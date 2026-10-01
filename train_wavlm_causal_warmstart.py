"""Train causal WavLM-conditioned Vocos after importing official Vocos backbone weights."""

import json
import os
from pathlib import Path

from pytorch_lightning.cli import LightningCLI

from vocos.warmstart import (
    configure_warmstart_trainability,
    load_wavlm_causal_checkpoint,
    warmstart_wavlm_causal_generator,
)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("VOCOS_WARMSTART_FREEZE_PRETRAINED must be one of 0/1, false/true, no/yes, or off/on.")


def main() -> None:
    source_checkpoint = os.environ.get("VOCOS_WARMSTART_SOURCE")
    report_path = os.environ.get("VOCOS_WARMSTART_REPORT")
    if not source_checkpoint or not report_path:
        raise RuntimeError("Set VOCOS_WARMSTART_SOURCE and VOCOS_WARMSTART_REPORT before launching warm-start training.")
    freeze_pretrained = _parse_bool(os.environ.get("VOCOS_WARMSTART_FREEZE_PRETRAINED", "0"))
    source_kind = os.environ.get("VOCOS_WARMSTART_SOURCE_KIND", "official_mel")

    cli = LightningCLI(run=False)
    if source_kind == "official_mel":
        report = warmstart_wavlm_causal_generator(cli.model, source_checkpoint)
    elif source_kind == "causal_wavlm":
        report = load_wavlm_causal_checkpoint(cli.model, source_checkpoint)
    else:
        raise ValueError("VOCOS_WARMSTART_SOURCE_KIND must be 'official_mel' or 'causal_wavlm'.")
    report["source_kind"] = source_kind
    report["trainability"] = configure_warmstart_trainability(cli.model, freeze_pretrained)
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Warm-started {report['loaded_target_key_count']} tensors from {source_checkpoint} ({source_kind}).")
    print(f"Freeze transferred backbone parameters: {freeze_pretrained}")
    print(f"Warm-start report: {report_destination}")
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
