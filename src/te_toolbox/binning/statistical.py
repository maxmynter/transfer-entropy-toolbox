"""Statistical and distribution based discretization methods."""

import logging
from collections.abc import Callable
from math import lgamma, log

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def optimize_bins(
    data: npt.NDArray[np.float64],
    cost_function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    minimize: bool = True,
    patience: int = 10,
) -> npt.NDArray[np.float64]:
    """
    Find optimal binning by scanning until cost function stops improving.

    Uses early stopping with patience to avoid unnecessary computation.

    Args:
    ----
        data: Input array
        cost_function: Function that computes cost for given histogram and bins
        minimize: If True, look for minimum. If False, look for maximum
        patience: Number of consecutive worse results before stopping

    Returns:
    -------
        Optimal bin edges

    """
    n = len(data)
    n_min = max(2, int(np.sqrt(n) * 0.1))  # Start with a reasonable minimum
    n_max = min(n, int(np.sqrt(n) * 10))  # Upper limit as safety

    best_cost = float("inf") if minimize else float("-inf")
    last_cost = best_cost
    best_n = n_min
    worse_count = 0

    for n_bins in range(n_min, n_max):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        hist, _ = np.histogram(data, bins)
        cost = cost_function(hist, bins)

        # Check if we found a better solution
        if (minimize and cost < best_cost) or (not minimize and cost > best_cost):
            best_cost = cost
            best_n = n_bins
            worse_count = 0
        elif (minimize and cost < last_cost) or (not minimize and cost > last_cost):
            worse_count = 0
        else:
            worse_count += 1
            if worse_count >= patience:
                break
    else:
        logger.warning(
            f"Warning: reached maximum_bins ({n_max}) without identifying optimum."
        )

    return np.linspace(np.min(data), np.max(data), best_n)


def knuth_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """
    Knuth's rule cost function (to be maximized).

    Args:
    ----
        hist: Histogram counts
        bins: Bin edges

    Returns:
    -------
        Cost value to be maximized

    """
    n = np.sum(hist)
    m = len(bins) - 1
    return float(
        n * log(m)
        + lgamma(m / 2)
        - lgamma(n + m / 2)
        - m * lgamma(0.5)
        + sum(lgamma(count + 0.5) for count in hist)
    )


def shimazaki_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """
    Shimazaki-Shinomoto cost function (to be minimized).

    Args:
    ----
        hist: Histogram counts
        bins: Bin edges

    Returns:
    -------
        Cost value to be minimized

    """
    n = np.sum(hist)
    h = bins[1] - bins[0]
    mean = np.mean(hist)
    var = np.var(hist)
    return float((2 * mean - var) / (h * n) ** 2)


def aic_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """
    AIC cost function (to be minimized).

    Args:
    ----
        hist: Histogram counts
        bins: Bin edges

    Returns:
    -------
        Cost value to be minimized

    """
    n = np.sum(hist)
    m = len(hist)
    h = bins[1] - bins[0]
    nonzero_hist = hist[hist > 0]
    return float(
        m + n * np.log(n) + n * np.log(h) - np.sum(nonzero_hist * np.log(nonzero_hist))
    )


def aicc_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """AIC cost function with small sample correction (to be minimized)."""
    n = np.sum(hist)
    m = len(bins) - 1

    aic = aic_cost(hist, bins)
    if n > m + 1:
        correction = 2 * m * (m + 1) / (n - m - 1)
        return float(aic + correction)
    return float("inf")


def bic_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """
    BIC cost function (to be minimized).

    Args:
    ----
        hist: Histogram counts
        bins: Bin edges

    Returns:
    -------
        Cost value to be minimized

    """
    n = np.sum(hist)
    h = bins[1] - bins[0]
    m = len(hist)
    nonzero_hist = hist[hist > 0]
    return float(
        np.log(n) / 2 * m
        + n * np.log(n)
        + n * np.log(h)
        - np.sum(nonzero_hist * np.log(nonzero_hist))
    )


def knuth_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using Knuth's rule.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, knuth_cost, minimize=False)


def shimazaki_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using Shimazaki-Shinomoto method.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, shimazaki_cost, minimize=True)


def aic_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using AIC.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, aic_cost, minimize=True)


def bic_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using BIC.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, bic_cost, minimize=True)


def small_sample_akaike_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using AIC with small sample correction.

    Uses the corrected AIC formula: AICc = AIC + 2k(k+1)/(n-k-1)
    where k is the number of bins and n is the sample size.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, aicc_cost, minimize=True)
