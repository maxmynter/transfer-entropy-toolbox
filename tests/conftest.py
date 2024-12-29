"""Utilities for tests."""

import numpy as np
import numpy.typing as npt


def bin_generator(data: npt.NDArray, n: int):
    """Generate non-zero width bins covering the data span."""
    # If 2D data, generate bins for each column
    if data.ndim > 1:
        return [
            np.linspace(np.min(data[:, i]) - 1e-6, np.max(data[:, i]) + 1e-6, n)
            for i in range(data.shape[1])
        ]

    # For 1D data, generate a single set of bins
    return np.linspace(np.min(data) - 1e-6, np.max(data) + 1e-6, n)
