"""Conditional Entropy."""

from typing import overload

import numpy as np

from ..core import MATRIX_DIMS
from ..core.types import BinType, FloatArray
from ..univariate.entropy import entropy
from .joint_entropy import joint_entropy


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
