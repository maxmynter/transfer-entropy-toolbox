"""Transfer entropy defnition and branched by Python or C++ backend."""

import numpy as np
from typing_extensions import overload

from te_toolbox.entropies.core.backend import Backend, get_backend

from . import cpp
from .core.discretization import _discretize_nd_data
from .core.types import BinType, FloatArray, IntArray
from .python.transfer.base import (
    discrete_transfer_entropy as python_discrete_transfer_entropy,
)


@overload
def _cpp_discrete_transfer_entropy(
    data: IntArray, n_classes: int | list[int], lag: int, at: tuple[int, int]
) -> np.float64: ...
@overload
def _cpp_discrete_transfer_entropy(
    data: IntArray, n_classes: int | list[int], lag: int, at: None
) -> FloatArray: ...
@overload
def _cpp_discrete_transfer_entropy(
    data: IntArray, n_classes: int | list[int], lag: int
) -> FloatArray: ...


def _cpp_discrete_transfer_entropy(
    classes: IntArray,
    n_classes: int | list[int],
    lag: int,
    at: tuple[int, int] | None = None,
) -> FloatArray | np.float64:
    dim = classes.shape[1]
    if isinstance(n_classes, int):
        n_classes = [n_classes] * dim
    if at is not None:
        return cpp.discrete_transfer_entropy(classes, n_classes, lag)
    tent = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                continue
            tent[i, j] = cpp.discrete_transfer_entropy(
                np.column_stack([classes[:, i], classes[:, j]]), n_classes, lag
            )
    return tent


def transfer_entropy(
    data: FloatArray, bins: BinType, lag: int, at: tuple[int, int] | None = None
) -> FloatArray | np.float64:
    """Calculate transfer entropy between all pairs of variables or single if at is set.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram.
        lag: Time lag for analysis.
        at: Tuple of index pair if only that index should be computed.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing transfer
            entropy values. Entry [i,j] is the transfer entropy from X_j to X_i.
        float: if at is set

    """
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
            return python_discrete_transfer_entropy(indices, n_classes, lag, at)
        case Backend.CPP:
            return _cpp_discrete_transfer_entropy(indices, n_classes, lag, at)
        case _:
            raise ValueError("Unkown Backend")
