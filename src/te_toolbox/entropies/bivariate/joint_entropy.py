"""Joint entropy."""

from typing import overload

import numpy as np
from numba import jit

from ..core import MATRIX_DIMS
from ..core.discretization import _discretize_nd_data
from ..core.types import BinType, FloatArray, IntArray, NClasses


@jit(nopython=True, fastmath=True)
def _discrete_bivariate_joint_entropy(
    data: IntArray, n_classes: list[int], at: tuple[int, int]
) -> np.float64:
    i, j = at
    n_steps, _ = data.shape
    hist = np.zeros((n_classes[i], n_classes[j]))

    for k in range(n_steps):
        hist[data[k, i], data[k, j]] += 1

    p_xy = hist / n_steps
    entropy_sum = 0.0
    for ii in range(p_xy.shape[0]):
        for jj in range(p_xy.shape[1]):
            if p_xy[ii, jj] > 0:
                entropy_sum += p_xy[ii, jj] * np.log(p_xy[ii, jj])

    return np.float64(-entropy_sum)


@overload
def discrete_joint_entropy(
    data: IntArray,
    n_classes: NClasses,
) -> FloatArray: ...


@overload
def discrete_joint_entropy(
    data: IntArray,
    n_classes: NClasses,
    at: None,
) -> FloatArray: ...


@overload
def discrete_joint_entropy(
    data: IntArray,
    n_classes: int | list[int],
    at: tuple[int, int],
) -> np.float64: ...


def discrete_joint_entropy(
    data: IntArray,
    n_classes: int | list[int],
    at: tuple[int, int] | None = None,
) -> FloatArray | np.float64:
    """Calculate the pairwise discrete joint entropy from class assignments."""
    if data.ndim != MATRIX_DIMS:
        raise ValueError(
            "Need 2 dimensional array [timesteps x variables] "
            "to calculate pairwise discrete joint entropy."
        )
    _, n_vars = data.shape
    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars

    if at is not None:
        return np.float64(_discrete_bivariate_joint_entropy(data, n_classes, at))

    jent = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        for j in range(i, n_vars):
            jent[i, j] = jent[j, i] = _discrete_bivariate_joint_entropy(
                data, n_classes, (i, j)
            )
    return jent


@overload
def joint_entropy(
    data: FloatArray,
    bins: BinType,
    at: tuple[int, int],
) -> np.float64: ...


@overload
def joint_entropy(
    data: FloatArray,
    bins: BinType,
) -> FloatArray: ...


@overload
def joint_entropy(
    data: FloatArray,
    bins: BinType,
    at: None,
) -> FloatArray: ...


def joint_entropy(
    data: FloatArray,
    bins: BinType,
    at: tuple[int, int] | None = None,
) -> FloatArray | np.float64:
    """Calculate pairwise joint entropy between all variables in the dataset.

    Args:
    ----
        data: Input data array of shape [timesteps x variables].
        bins: Number of bins or bin edges. Can be:
            - int: Same number of bins for all variables
            - [int, int]: Different number of bins for each variable
            - array: Bin edges for all variables
            - [array, array]: Different bin edges for each variable
        at: Tuple of index pair if only that index should be computed.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing joint entropies.
            Entry [i,j] is the joint entropy H(X_i, X_j).
        float: If at parameter is set

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim != MATRIX_DIMS:
        raise ValueError("Data must be 2-dimensional [timesteps x variables]")
    _, n_vars = data.shape

    if isinstance(bins, int | np.ndarray):
        bins = [bins] * n_vars

    # Convert to tuples for hashing
    data_tuple = tuple(tuple(data[:, i]) for i in range(n_vars))
    bins_tuple = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuple, bins_tuple)
    indices = np.column_stack([d[0] for d in discretized])
    n_classes = [d[1] for d in discretized]
    return discrete_joint_entropy(indices, n_classes, at)
