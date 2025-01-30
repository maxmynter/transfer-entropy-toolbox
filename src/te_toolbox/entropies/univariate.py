"""Univariate Entropy."""

from typing import overload

import numpy as np

from . import cpp
from .core import MATRIX_DIMS, VECTOR_DIMS, Backend, get_backend
from .core.discretization import _discretize_nd_data
from .core.types import BinType, FloatArray, IntArray
from .python import univariate as python_univariate


@overload
def _cpp_discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int
) -> np.float64: ...


@overload
def _cpp_discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: None
) -> FloatArray: ...


@overload
def _cpp_discrete_entropy(data: IntArray, n_classes: int | list[int]) -> FloatArray: ...


def _cpp_discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int | None = None
) -> FloatArray | np.float64:
    if data.ndim == 1:
        return np.float64(cpp.discrete_entropy(data, n_classes))

    _, n_vars = data.shape

    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars

    if at is not None:
        return np.float64(
            cpp.discrete_entropy(np.ravel(data[:, at]), int(n_classes[at]))
        )
    else:
        probs = np.empty(n_vars, dtype=np.float64)
        for i in range(n_vars):
            probs[i] = cpp.discrete_entropy(np.ravel(data[:, i]), int(n_classes[i]))
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

    match get_backend():
        case Backend.PY:
            return python_univariate.discrete_entropy(indices, n_classes, at)
        case Backend.CPP:
            return _cpp_discrete_entropy(indices, n_classes, at)
        case _:
            raise ValueError("Unkown Backend")
