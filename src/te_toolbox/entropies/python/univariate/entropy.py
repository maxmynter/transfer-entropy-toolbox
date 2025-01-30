"""Univariate Entropies."""

from typing import overload

import numpy as np
from numba import jit

from ...core.types import FloatArray, IntArray


@jit(nopython=True, fastmath=True)
def _discrete_univariate_entropy(
    data: IntArray, n_classes: list[int], at: int
) -> np.float64:
    n_steps = data.shape[0]
    p = np.bincount(data[:, at], minlength=n_classes[at]) / n_steps
    nonzero = p > 0
    return np.float64(-np.sum(p[nonzero] * np.log(p[nonzero])))


@overload
def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int
) -> np.float64: ...


@overload
def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: None
) -> FloatArray: ...


@overload
def discrete_entropy(
    data: IntArray,
    n_classes: int | list[int],
) -> FloatArray: ...


def discrete_entropy(
    data: IntArray, n_classes: int | list[int], at: int | None = None
) -> FloatArray | np.float64:
    """Calculate the discrete entropy from class assignments."""
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    _, n_vars = data.shape

    if isinstance(n_classes, int):
        n_classes = [n_classes] * n_vars

    if at is not None:
        return np.float64(_discrete_univariate_entropy(data, n_classes, at))
    else:
        probs = np.empty(n_vars, dtype=np.float64)
        for i in range(n_vars):
            probs[i] = _discrete_univariate_entropy(data, n_classes, i)

        return probs
