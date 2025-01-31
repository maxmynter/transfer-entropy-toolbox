"""Multivariate Joint Entropy."""

import numpy as np

from te_toolbox.entropies.core.backend import Backend
from te_toolbox.entropies.utils import branch_funcs_by_backends

from . import cpp
from .core.discretization import _discretize_nd_data
from .core.types import BinType, FloatArray
from .python.multivariates import (
    discrete_multivar_joint_entropy as python_discrete_multivar_joint_entropy,
)


def multivar_joint_entropy(
    data: FloatArray,
    bins: BinType,
) -> np.float64:
    """Calculate joint entropy for n variables.

    Args:
    ----
        data: Input array of shape [timesteps x variables].
        bins: Number of bins for all, for each, or list of bin edges for histogram.

    Returns:
    -------
        float: Joint entropy value H(X1,...,Xn).

    """
    if isinstance(bins, int | np.ndarray):
        bins = [bins] * data.shape[1]

    # Convert data and bins to hashable tuples for caching
    data_tuples = tuple(tuple(col) for col in data.T)
    bins_tuples = tuple(b if isinstance(b, int) else tuple(b) for b in bins)

    discretized = _discretize_nd_data(data_tuples, bins_tuples)
    classes = [d[0] for d in discretized]
    n_classes = [d[1] for d in discretized]

    return branch_funcs_by_backends(
        {
            Backend.PY: python_discrete_multivar_joint_entropy,
            Backend.CPP: cpp.discrete_multivar_joint_entropy,
        },
        classes,
        n_classes,
    )
