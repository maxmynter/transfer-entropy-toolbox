"""Canonical transfer entropies."""

import numpy as np

from ...core.types import FloatArray, IntArray
from ..bivariate import discrete_joint_entropy
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
            if i == j:
                continue
            h_y_ylag_xlag = discrete_multivar_joint_entropy(
                [current[:, i], lagged[:, i], lagged[:, j]],
                [n_classes[i], n_classes[i], n_classes[j]],
            )
            tent[i, j] = h_y_ylag + h_xy_lag[i, j] - h_y_ylag_xlag - h_x_lag[i]
    return tent
