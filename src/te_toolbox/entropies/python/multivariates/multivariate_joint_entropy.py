"""Multivariate joint entropy."""

import numpy as np

from ...core.discretization import _discretize_nd_data
from ...core.types import BinType, FloatArray, IntArray


def discrete_multivar_joint_entropy(
    classes: list[IntArray],
    n_classes: list[int],
) -> np.float64:
    """Calculate joint entropy from discrete classes for multiple variables."""
    n_steps = classes[0].shape
    hist = np.zeros(n_classes)

    idx = tuple(c for c in classes)
    np.add.at(hist, idx, 1)

    p = hist / n_steps
    nonzero_mask = p > 0
    return np.float64(-np.sum(p[nonzero_mask] * np.log(p[nonzero_mask])))


def multivar_joint_entropy(
    data: FloatArray,
    bins: BinType,
) -> np.float64:
    """Calculate joint entropy for n variables.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins for all, for each, or list of bin edges for histogram.

    Returns:
    -------
        float: Joint entropy value H(X1,...,Xn).

    """
    if isinstance(bins, int | np.ndarray):
        bins = [bins] * data.shape[1]

    # Convert data and bins to hashable tuples for caching
    data_tuples = tuple(tuple(col) for col in data.T)
    bins_tuples = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuples, bins_tuples)
    classes = [d[0] for d in discretized]
    n_classes = [d[1] for d in discretized]

    return discrete_multivar_joint_entropy(classes, n_classes)
