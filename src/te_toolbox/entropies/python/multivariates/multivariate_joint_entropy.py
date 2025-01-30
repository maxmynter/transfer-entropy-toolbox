"""Multivariate joint entropy."""

import numpy as np

from ...core.types import IntArray


def discrete_multivar_joint_entropy(
    classes: list[IntArray],
    n_classes: list[int],
) -> np.float64:
    """Calculate joint entropy from discrete classes for multiple variables."""
    n_steps = classes[0].shape
    hist = np.zeros(n_classes)

    idx = tuple(c for c in classes)
    np.add.at(hist, idx, 1)

    p = hist / n_steps
    nonzero_mask = p > 0
    return np.float64(-np.sum(p[nonzero_mask] * np.log(p[nonzero_mask])))
