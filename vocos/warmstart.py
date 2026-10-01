"""Transfer compatible generator weights from the official mel Vocos checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _target_key(source_key: str) -> str | None:
    """Map a non-causal Vocos backbone key to its causal counterpart."""
    if not source_key.startswith("backbone."):
        return None
    if source_key.startswith("backbone.embed."):
        return source_key.replace("backbone.embed.", "backbone.embed.conv.", 1)
    return source_key.replace(".dwconv.", ".dwconv.conv.")


def warmstart_wavlm_causal_generator(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load the shape-compatible official Vocos generator weights into ``model``.

    The source checkpoint is mel-conditioned and non-causal.  ConvNeXt block
    weights and normalization parameters have compatible shapes after mapping
    the causal convolution wrappers.  The 100-channel mel input stem is
    converted to a 1024-channel WavLM stem by averaging its input filters,
    repeating them, and variance-scaling the result.  The source ISTFT head,
    feature extractor, and all discriminators are intentionally left random:
    their shapes or causal semantics differ.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    loaded: list[str] = []
    skipped: dict[str, str] = {}
    for source_key, source_tensor in source_state.items():
        target_key = _target_key(source_key)
        if target_key is None:
            skipped[source_key] = "outside_generator_backbone"
            continue
        if target_key not in target_state:
            skipped[source_key] = "missing_target_key"
            continue
        if source_tensor.shape != target_state[target_key].shape:
            skipped[source_key] = f"shape_mismatch:{tuple(source_tensor.shape)}->{tuple(target_state[target_key].shape)}"
            continue
        target_state[target_key] = source_tensor
        loaded.append(target_key)

    source_stem_key = "backbone.embed.weight"
    target_stem_key = "backbone.embed.conv.weight"
    if source_stem_key not in source_state or target_stem_key not in target_state:
        raise KeyError("The source or target input stem is missing.")
    source_stem = source_state[source_stem_key]
    target_stem = target_state[target_stem_key]
    if source_stem.ndim != 3 or target_stem.ndim != 3 or source_stem.shape[0] != target_stem.shape[0] or source_stem.shape[2] != target_stem.shape[2]:
        raise ValueError(f"Cannot adapt input stem {tuple(source_stem.shape)} to {tuple(target_stem.shape)}.")
    repeated_stem = source_stem.mean(dim=1, keepdim=True).expand_as(target_stem).clone()
    repeated_stem.mul_((source_stem.size(1) / target_stem.size(1)) ** 0.5)
    target_state[target_stem_key] = repeated_stem
    loaded.append(target_stem_key)

    model.load_state_dict(target_state, strict=True)
    return {
        "source_checkpoint": str(checkpoint_path),
        "loaded_target_keys": sorted(loaded),
        "loaded_target_key_count": len(loaded),
        "skipped_source_keys": skipped,
        "stem_initialization": "mean-over-mel-channels, repeat-to-WavLM-channels, variance-scaled",
        "uninitialized_components": ["feature_extractor", "head", "multiperioddisc", "multiresddisc"],
    }


def load_wavlm_causal_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load matching weights from a previous causal WavLM Vocos checkpoint.

    This intentionally copies model weights only. Optimizer and scheduler state
    are discarded so a subsequent full fine-tune can use new learning rates.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    loaded: list[str] = []
    skipped: dict[str, str] = {}
    for source_key, source_tensor in source_state.items():
        if source_key not in target_state:
            skipped[source_key] = "missing_target_key"
            continue
        if source_tensor.shape != target_state[source_key].shape:
            skipped[source_key] = f"shape_mismatch:{tuple(source_tensor.shape)}->{tuple(target_state[source_key].shape)}"
            continue
        target_state[source_key] = source_tensor
        loaded.append(source_key)

    model.load_state_dict(target_state, strict=True)
    return {
        "source_checkpoint": str(checkpoint_path),
        "loaded_target_keys": sorted(loaded),
        "loaded_target_key_count": len(loaded),
        "skipped_source_keys": skipped,
        "source_type": "causal_wavlm_checkpoint",
        "optimizer_state_loaded": False,
    }


def configure_warmstart_trainability(model: torch.nn.Module, freeze_pretrained: bool) -> dict[str, Any]:
    """Choose whether transferred backbone parameters remain trainable.

    When frozen, the adapted WavLM input stem and newly initialized causal head
    and discriminators remain trainable.  All transferred ConvNeXt backbone
    blocks and their normalization layers are frozen.
    """
    trainable_prefixes = (
        "backbone.embed.",  # Adapted from the 100-channel mel input stem.
        "head.",  # New causal 480-sample-hop ISTFT head.
        "multiperioddisc.",
        "multiresddisc.",
    )
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not freeze_pretrained or name.startswith(trainable_prefixes))
        (trainable_names if parameter.requires_grad else frozen_names).append(name)

    return {
        "freeze_pretrained": freeze_pretrained,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
    }
