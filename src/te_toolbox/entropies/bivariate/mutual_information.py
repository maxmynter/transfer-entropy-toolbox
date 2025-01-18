"""Mutual Information."""

import numpy as np

from ..core.types import BinType, FloatArray
from ..univariate import entropy
from .joint_entropy import joint_entropy


def mutual_information(
    data: FloatArray,
    bins: BinType,
    norm: bool = True,
) -> FloatArray:
    """Calculate mutual information of time series.

    Args:
    ----
        data: Input data array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram approximation of PDF.
        norm: Whether to normalize result between 0 and 1 using I(X,Y)/sqrt(H(X)*H(Y)).

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing
            mutual information values. Entry [i,j] is I(X_i,X_j).

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim == 1:
        raise ValueError("Cannot compute mutual information with single variable.")

    dim = data.shape[1]
    h_xy = joint_entropy(data, bins)
    h_x = entropy(data, bins)

    i, j = np.meshgrid(range(dim), range(dim))
    idx = i.astype(np.int64)
    jdx = j.astype(np.int64)

    mi = np.float64(h_x[idx] + h_x[jdx] - h_xy[idx, jdx])
    if norm:
        denominator = np.float64(np.sqrt(np.multiply(h_x[i], h_x[j])))
        if np.any(denominator == 0):
            raise ValueError("Zero entropy, cannot normalize mutual information.")
        mi = np.divide(mi, denominator)
    return mi
