"""Normalized transfer entropies."""

from typing import overload

import numpy as np

from ..bivariate import discrete_joint_entropy
from ..core.discretization import _discretize_nd_data
from ..core.types import BinType, FloatArray, IntArray, NClasses
from ..multivariates import discrete_multivar_joint_entropy
from ..univariate import discrete_entropy
from .base import discrete_transfer_entropy


def normalized_transfer_entropy(
    data: FloatArray, bins: BinType, lag: int, at: tuple[int, int] | None = None
) -> FloatArray | np.float64:
    """Calculate normalized transfer entropy between variables.

    Normalized as: 1 - H(Y_t | Y_t_lag, X_t_lag) / H(Y_t | Y_t_lag)
    where H(Y_t | Y_t_lag, X_t_lag) = H(Y_t, Y_t_lag, X_t_lag) - H(Y_t_lag, X_t_lag)
    and H(Y_t | Y_t_lag) = H(Y_t, Y_t_lag) - H(Y_t_lag).

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins or bin edges for histogram.
        lag: Time lag for analysis.
        at: Tuple of index pair if only that index should be computed.

    Returns:
    -------
        ndarray: Matrix of shape [n_variables, n_variables] containing normalized
            transfer entropy values. Entry [i,j] is between 0 and 1.
        float: if at is set

    Raises:
    ------
        ValueError: If data dimensions are invalid.

    """
    n_steps, n_vars = data.shape
    if isinstance(bins, int | np.ndarray):
        bins = [bins] * n_vars

    data_tuple = tuple(tuple(data[:, i]) for i in range(n_vars))
    bins_tuple = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuple, bins_tuple)
    indices = np.column_stack([d[0] for d in discretized])
    n_classes = [d[1] for d in discretized]

    return discrete_normalized_transfer_entropy(indices, n_classes, lag, at)


@overload
def discrete_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: tuple[int, int]
) -> np.float64: ...


@overload
def discrete_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: None
) -> FloatArray: ...


@overload
def discrete_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int
) -> FloatArray: ...


def discrete_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: tuple[int, int] | None = None
) -> FloatArray | np.float64:
    """Calculate H-normalized transfer entropy between discrete variables.

    Normalized as: 1 - H(Y_t | Y_t_lag, X_t_lag) / H(Y_t | Y_t_lag)

    Args:
    ----
        classes: Array of discrete state indices [timesteps x variables]
        n_classes: Number of bins for each variable
        lag: Time lag for analysis
        at: Tuple of index pair if only that index should be computed.

    Returns:
    -------
        Matrix containing normalized transfer entropy values.
        Entry [i,j] is normalized TE from X_j to X_i.
        Float if at is set.

    """
    n_steps, n_vars = classes.shape

    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars

    current = classes[lag:]
    lagged = classes[:-lag]

    ntent = np.zeros((n_vars, n_vars))

    h_xy_lag = discrete_joint_entropy(lagged, n_classes)
    h_x_lag = discrete_entropy(lagged, n_classes)

    if at is not None:
        i, j = at
        h_y_ylag_at = discrete_joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            [n_classes[i], n_classes[i]],
            at=(i, j),
        )
        h_y_ylag_xlag_at = discrete_multivar_joint_entropy(
            [current[:, i], lagged[:, i], lagged[:, j]],
            [n_classes[i], n_classes[i], n_classes[j]],
        )
        h_y_given_ylag_xlag_at = h_y_ylag_xlag_at - h_xy_lag[i, j]
        h_y_given_ylag_at = h_y_ylag_at - h_x_lag[i]
        ntent_ij = (
            1 - h_y_given_ylag_xlag_at / h_y_given_ylag_at
            if h_y_given_ylag_at != 0
            else 0
        )
        return np.float64(ntent_ij)

    for i in range(n_vars):
        _at = (0, 1)
        h_y_ylag = discrete_joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            [n_classes[i], n_classes[i]],
            at=_at,
        )
        for j in range(n_vars):
            h_y_ylag_xlag = discrete_multivar_joint_entropy(
                [current[:, i], lagged[:, i], lagged[:, j]],
                [n_classes[i], n_classes[i], n_classes[j]],
            )
            h_y_given_ylag_xlag = h_y_ylag_xlag - h_xy_lag[i, j]
            h_y_given_ylag = h_y_ylag - h_x_lag[i]
            ntent[i, j] = (
                1 - h_y_given_ylag_xlag / h_y_given_ylag if h_y_given_ylag != 0 else 0
            )
    return ntent


@overload
def discrete_logn_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: tuple[int, int]
) -> np.float64: ...


@overload
def discrete_logn_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: None
) -> FloatArray: ...


@overload
def discrete_logn_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int
) -> FloatArray: ...


def discrete_logn_normalized_transfer_entropy(
    classes: IntArray, n_classes: NClasses, lag: int, at: tuple[int, int] | None = None
) -> FloatArray | np.float64:
    """Calculate transfer entropy normalized by log(N) between discrete variables.

    Args:
    ----
        classes: Array of discrete state indices [timesteps x variables]
        n_classes: Number of bins for each variable
        lag: Time lag for analysis
        at: Tuple of index pair if only that index should be computed

    Returns:
    -------
        Matrix containing logN-normalized transfer entropy values.
        Entry [i,j] is normalized TE from X_j to X_i.
        Float if at is set.

    """
    te = discrete_transfer_entropy(classes, n_classes, lag, at)

    if isinstance(te, float):
        if isinstance(at, tuple):
            return np.float64(
                te / np.log(n_classes[at[0]])
                if isinstance(n_classes, list)
                else te / np.log(n_classes)
            )
        else:
            raise ValueError("Entropy returned float without 'at' parameter set")

    if isinstance(n_classes, int):
        te /= np.log(n_classes)
    else:
        for i in range(te.shape[0]):
            te[i] /= np.log(n_classes[i])

    return te


@overload
def logn_normalized_transfer_entropy(
    data: FloatArray, bins: BinType, lag: int, at: tuple[int, int]
) -> np.float64: ...


@overload
def logn_normalized_transfer_entropy(
    data: FloatArray, bins: BinType, lag: int, at: None
) -> FloatArray: ...


@overload
def logn_normalized_transfer_entropy(
    data: FloatArray, bins: BinType, lag: int
) -> FloatArray: ...


def logn_normalized_transfer_entropy(
    data: FloatArray, bins: BinType, lag: int, at: tuple[int, int] | None = None
) -> FloatArray | np.float64:
    """Calculate transfer entropy normalized by log(N) where N is number of bins.

    Args:
    ----
        data: Input array of shape [timesteps x variables]
        bins: Number of bins or bin edges for histogram
        lag: Time lag for analysis
        at: Tuple of index pair if only that index should be computed

    Returns:
    -------
        Matrix containing logN-normalized transfer entropy values.
        Entry [i,j] represents normalized TE from X_j to X_i.
        Float if at is set.

    """
    n_vars = data.shape[1]

    if isinstance(bins, int | np.ndarray):
        bins = [bins] * n_vars

    # Convert to tuples for caching
    data_tuple = tuple(tuple(data[:, i]) for i in range(n_vars))
    bins_tuple = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuple, bins_tuple)
    indices = np.column_stack([d[0] for d in discretized])
    n_classes = [d[1] for d in discretized]

    return discrete_logn_normalized_transfer_entropy(indices, n_classes, lag, at)
