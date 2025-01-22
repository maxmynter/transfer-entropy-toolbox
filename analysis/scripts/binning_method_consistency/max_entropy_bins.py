"""Entropy maximising Binning."""

import numpy as np

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)


def max_tent(tent, data: np.ndarray, lag=1):
    """Get tent maximising bins."""
    data = data.reshape(-1, 2)
    last_te = -float("inf")
    max_bins = np.array([])
    data_min = np.min(data)
    data_max = np.max(data)

    for n_edges in range(3, len(data)):
        bins = np.linspace(data_min, data_max, n_edges)
        te = tent(data, bins, lag, at=(1, 0))
        if te > last_te:
            max_bins = bins
            last_te = te
        else:
            return max_bins
    return max_bins


def max_tent_bins(data):
    """Get transfer entropy maximising bins."""
    return max_tent(transfer_entropy, data)


def max_ntent_bins(data):
    """Get H-normalised transfer entropy maximising bins."""
    return max_tent(normalized_transfer_entropy, data)


def max_logntent_bins(data):
    """Get log m transfer entropy maximizing bins."""
    return max_tent(logn_normalized_transfer_entropy, data)
