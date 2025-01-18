"""Discretization utilities for continuous time series."""

from functools import lru_cache

import numpy as np

from .types import IntArray


@lru_cache(maxsize=512)
def _discretize_1d_data(
    data: tuple[np.float64], bins: int | tuple[float, ...]
) -> tuple[IntArray, int]:
    """Convert dataset into discrete classes.

    Note: Values equal to outer bin edges handling:
    - Values equal to leftmost edge go into first bin
    - Values equal to rightmost edge go into last bin
    """
    data_array = np.array(data)
    if isinstance(bins, int):
        edges = np.linspace(data_array.min(), data_array.max(), bins + 1)
    else:
        if data_array.min() < bins[0] or data_array.max() > bins[-1]:
            raise ValueError("Data contains values outside of specified bin range")
        edges = bins

    # Subtract 1 as digitize is 1 indexed
    indices = np.digitize(data_array, edges, right=False) - 1

    n_bins = len(edges) - 1

    # Make rightmost bin edge inclusive
    indices = np.where(indices == n_bins, n_bins - 1, indices)
    return indices, len(edges) - 1


@lru_cache(maxsize=512)
def _discretize_nd_data(
    data_tuple: tuple[tuple[float, ...], ...],
    bins_tuple: tuple[int | tuple[float, ...], ...],
) -> tuple[tuple[IntArray, int], ...]:
    """Convert multiple variables into discrete classes.

    Args:
    ----
        data_tuple: Tuple of data arrays, one per variable
        bins_tuple: Tuple of bin specifications, one per variable

    Returns:
    -------
        Tuple of (indices, n_bins) pairs, one per variable

    """
    return tuple(
        _discretize_1d_data(d, b) for d, b in zip(data_tuple, bins_tuple, strict=True)
    )
