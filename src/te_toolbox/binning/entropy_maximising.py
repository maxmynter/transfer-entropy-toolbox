"""Entropy maximising binning."""

import numpy as np

from ..entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)
from .statistical import optimize_bins


def max_tent(tent, data: np.ndarray, lag=1, at: tuple[int, int] = (0, 1)):
    """Get tent maximising bins using binning optimizer."""
    data_2d = data.reshape(-1, 2)

    def cost(_, bins):
        try:
            return tent(data_2d, bins, lag, at=at)
        except Exception:
            print("Error calculating Max Entropy")
            return float("inf")

    tent_maximising_bins = optimize_bins(
        data=data_2d.flatten(),
        cost_function=cost,
        minimize=False,
        window_size=20,
        trend_patience=10,
        stationary_threshold=1e-4,
        method=tent.__name__,
    )
    return tent_maximising_bins


def max_tent_bins(data):
    """Get transfer entropy maximising bins."""
    return max_tent(transfer_entropy, data)


def max_ntent_bins(data):
    """Get H-normalised transfer entropy maximising bins."""
    return max_tent(normalized_transfer_entropy, data)


def max_logntent_bins(data):
    """Get log m transfer entropy maximizing bins."""
    return max_tent(logn_normalized_transfer_entropy, data)
