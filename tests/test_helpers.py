import numpy as np
from matplotlib import pyplot as plt

from vocos.helpers import save_figure_to_numpy


def test_save_figure_to_numpy_returns_rgb_uint8_image():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    image = save_figure_to_numpy(fig)
    plt.close(fig)

    assert image.dtype == np.uint8
    assert image.ndim == 3
    assert image.shape[-1] == 3
