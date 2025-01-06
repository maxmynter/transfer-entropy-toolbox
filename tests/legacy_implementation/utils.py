"""Utils for consistency check with old Thesis code."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def prepare_causal_dependent_data(
    x: npt.NDArray[np.float64], func: Callable[[float], float], noise: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create a causally dependent data from input."""
    y = np.array([func(x[i - 1]) for i in range(1, len(x))])

    y += np.random.normal(loc=0, scale=noise, size=len(x) - 1)

    return x[1:], y
