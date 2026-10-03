"""Command-line entry point for training Vocos models from a LightningCLI config."""

import os

from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli_kwargs = {"run": False}
    # Resume jobs keep the original resolved config in their durable run
    # directory. Do not overwrite it merely to restore a checkpoint.
    if os.environ.get("VOCOS_RESUME_CKPT"):
        cli_kwargs["save_config_callback"] = None
    cli = LightningCLI(**cli_kwargs)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
