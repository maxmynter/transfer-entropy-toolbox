"""Univariate Entropies."""

from typing import overload

import numpy as np
from numba import jit

from ..core import MATRIX_DIMS, VECTOR_DIMS
from ..core.discretization import _discretize_nd_data
from ..core.types import BinType, FloatArray, IntArray


@jit(nopython=True, fastmath=True)
def _discrete_univariate_entropy(
    data: IntArray, n_classes: list[int], at: int
) -> np.float64:
    n_steps = data.shape[0]
    p = np.bincount(data[:, at], minlength=n_classes[at]) / n_steps
    nonzero = p > 0
    return np.float64(-np.sum(p[nonzero] * np.log(p[nonzero])))


@overload
def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int
) -> np.float64: ...


@overload
def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: None
) -> FloatArray: ...


@overload
def discrete_entropy(
    data: IntArray,
    n_classes: int | list[int],
) -> FloatArray: ...


def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int | None = None
) -> FloatArray | np.float64:
    """Calculate the discrete entropy from class assignments."""
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    _, n_vars = data.shape

    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars

    if at is not None:
        return np.float64(_discrete_univariate_entropy(data, n_classes, at))
    else:
        probs = np.empty(n_vars, dtype=np.float64)
        for i in range(n_vars):
            probs[i] = _discrete_univariate_entropy(data, n_classes, i)

        return probs


@overload
def entropy(
    data: FloatArray,
    bins: BinType,
    at: int,
) -> np.float64: ...


@overload
def entropy(
    data: FloatArray,
    bins: BinType,
) -> FloatArray: ...


@overload
def entropy(
    data: FloatArray,
    bins: BinType,
    at: None,
) -> FloatArray: ...


def entropy(
    data: FloatArray,
    bins: BinType,
    at: int | None = None,
) -> FloatArray | np.float64:
    """Calculate the entropy of one or more datasets.

    Args:
    ----
        data: Input data array. Can be 1D or 2D [timesteps x variables].
        bins: Number of bins for histogram or list of bin edges.
        at: index if only univariate entropy should be computed.

    Returns:
    -------
        float: Entropy value for 1D input
        ndarray: Array of entropy values for 2D input, one per variable

    Raises:
    ------
        ValueError: If data dimensions are invalid

    """
    if data.size == 0:
        raise ValueError("Cannot compute entropy of empty array")
    if data.ndim > MATRIX_DIMS or data.ndim < VECTOR_DIMS:
        raise ValueError(
            "Wrong data format."
            "Data must be of dimension [timesteps] or [timesteps x variables]"
        )

    data = data.reshape(-1, 1) if data.ndim == 1 else data
    n_vars = data.shape[1]

    if isinstance(bins, int | np.ndarray):
        bins = [bins] * n_vars

    data_tuple = tuple(tuple(data[:, i]) for i in range(n_vars))
    bins_tuple = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuple, bins_tuple)
    indices = np.column_stack([d[0] for d in discretized])
    n_classes = [d[1] for d in discretized]

    return discrete_entropy(indices, n_classes, at)
