"""Bivariate Entropies."""

from typing import overload

import numpy as np

from . import cpp
from .core import MATRIX_DIMS, Backend, get_backend
from .core.discretization import _discretize_nd_data
from .core.types import BinType, FloatArray, IntArray
from .python.bivariate.joint_entropy import (
    discrete_joint_entropy as python_discrete_joint_entropy,
)
from .univariate import entropy


@overload
def _cpp_discrete_joint_entropy(
    data: IntArray, n_classes: list | list[int], at: tuple[int, int]
) -> np.float64: ...


@overload
def _cpp_discrete_joint_entropy(
    data: IntArray,
    n_classes: list | list[int],
) -> FloatArray: ...


@overload
def _cpp_discrete_joint_entropy(
    data: IntArray, n_classes: list | list[int], at: None
) -> np.float64: ...


def _cpp_discrete_joint_entropy(
    data: IntArray, n_classes: list | list[int], at: tuple[int, int] | None = None
) -> np.float64 | FloatArray:
    _, n_vars = data.shape
    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars
    if at is not None:
        return cpp.discrete_joint_entropy(
            data[:, at], [n_classes[at[0]], n_classes[at[1]]]
        )
    jent = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i, n_vars):
            columns = np.column_stack([data[:, i], data[:, j]])
            jent[i, j] = jent[j, i] = cpp.discrete_joint_entropy(
                columns, [n_classes[i], n_classes[j]]
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

    match get_backend():
        case Backend.PY:
            return python_discrete_joint_entropy(indices, n_classes, at)
        case Backend.CPP:
            return _cpp_discrete_joint_entropy(indices, n_classes, at)
        case _:
            raise ValueError("Unkown Backend")


@overload
def conditional_entropy(
    data: FloatArray,
    bins: BinType,
    at: tuple[int, int],
) -> np.float64: ...


@overload
def conditional_entropy(
    data: FloatArray,
    bins: BinType,
) -> FloatArray: ...


@overload
def conditional_entropy(
    data: FloatArray,
    bins: BinType,
    at: None,
) -> FloatArray: ...


def conditional_entropy(
    data: FloatArray,
    bins: BinType,
    at: tuple[int, int] | None = None,
) -> FloatArray | np.float64:
    """Calculate conditional entropy between all pairs of variables.

    Uses the chain rule: H(Y|X) = H(X,Y) - H(X)

    Args:
    ----
        data: Input data array or DataFrame of shape [timesteps x variables].
        bins: Number of bins or bin edges (same formats as joint_entropy).
        at: Tuple of index pair if only that combination should be computed.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing
            conditional entropies. Entry [i,j] is the conditional entropy H(X_i|X_j).
        float: if at is set.

    Raises:
    ------
        ValueError: If data has invalid dimensions or single variable.

    """
    if data.ndim < MATRIX_DIMS:
        raise ValueError(
            "Need more than 2 time series to calculate conditional entropy"
        )
    if at is not None:
        univar_h_xy = joint_entropy(data, bins, at)
        univar_h_x = entropy(data, bins, at=at[1])

        return univar_h_xy - univar_h_x
    else:
        h_xy = joint_entropy(data, bins)
        h_x = entropy(data, bins)
        return (h_xy - h_x.reshape(1, -1)).astype(np.float64)
