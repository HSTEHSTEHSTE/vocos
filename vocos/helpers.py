import os
import signal
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback

matplotlib.use("Agg")


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Save a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure object.

    Returns:
        ndarray: Numpy array representing the figure.
    """
    fig.canvas.draw()
    # `tostring_rgb` was removed from FigureCanvasAgg in Matplotlib 3.10.
    # `buffer_rgba` is the supported API; copy before callers close the figure.
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class GradNormCallback(Callback):
    """
    Callback to log the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", gradient_norm(model))


class SlurmPreemptionCheckpointCallback(Callback):
    """Save a checkpoint and stop cleanly when Slurm warns of an upcoming limit."""

    def __init__(self, filename: str = "preempt.ckpt"):
        self.filename = filename
        self._checkpoint_requested = False
        self._previous_handler = None

    def setup(self, trainer, pl_module, stage=None):
        if not hasattr(signal, "SIGUSR1") or os.environ.get("SLURM_JOB_ID") is None:
            return
        self._previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._request_checkpoint)

    def teardown(self, trainer, pl_module, stage=None):
        if self._previous_handler is not None:
            signal.signal(signal.SIGUSR1, self._previous_handler)
            self._previous_handler = None

    def _request_checkpoint(self, signum, frame):
        self._checkpoint_requested = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._checkpoint_requested:
            return

        checkpoint_path = Path(trainer.default_root_dir) / "checkpoints" / self.filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
        trainer.should_stop = True
        self._checkpoint_requested = False


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Gradient norm.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm
