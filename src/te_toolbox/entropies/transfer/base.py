"""Canonical transfer entropies."""

import numpy as np

from ..bivariate import discrete_joint_entropy
from ..core.discretization import _discretize_nd_data
from ..core.types import BinType, FloatArray, IntArray
from ..multivariates import discrete_multivar_joint_entropy
from ..univariate import discrete_entropy


def discrete_transfer_entropy(
    classes: IntArray,
    n_classes: int | list[int],
    lag: int,
    at: tuple[int, int] | None = None,
) -> FloatArray | np.float64:
    """Calculate transfer entropy between all pairs of discrete variables.

    Args:
    ----
        classes: Array of discrete state indices [timesteps x variables]
        n_classes: Number of bins for each variable
        lag: Time lag for analysis
        at: Tuple of index pair if only that index should be computed

    Returns:
    -------
        Matrix containing transfer entropy values. Entry [i,j] is TE from X_j to X_i or
        Float if at is set

    """
    if isinstance(n_classes, int):
        n_classes = [n_classes] * classes.shape[1]

    current = classes[lag:]
    lagged = classes[:-lag]

    dim = current.shape[1]
    tent = np.zeros((dim, dim))

    h_xy_lag = discrete_joint_entropy(lagged, n_classes)
    h_x_lag = discrete_entropy(lagged, n_classes)

    if at is not None:
        i, j = at
        h_y_ylag_at = discrete_joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            [n_classes[i], n_classes[i]],
            at=(0, 1),
        )
        h_y_ylag_xlag_at = discrete_multivar_joint_entropy(
            [current[:, i], lagged[:, i], lagged[:, j]],
            [n_classes[i], n_classes[i], n_classes[j]],
        )
        tent_ij = h_y_ylag_at + h_xy_lag[i, j] - h_y_ylag_xlag_at - h_x_lag[i]
        return np.float64(tent_ij)

    for i in range(dim):
        h_y_ylag = discrete_joint_entropy(
            np.column_stack([current[:, i], lagged[:, i]]),
            [n_classes[i], n_classes[i]],
            at=(0, 1),
        )
        for j in range(dim):
            h_y_ylag_xlag = discrete_multivar_joint_entropy(
                [current[:, i], lagged[:, i], lagged[:, j]],
                [n_classes[i], n_classes[i], n_classes[j]],
            )
            tent[i, j] = h_y_ylag + h_xy_lag[i, j] - h_y_ylag_xlag - h_x_lag[i]
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

    return discrete_transfer_entropy(indices, n_classes, lag, at)
