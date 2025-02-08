"""Statistical and distribution based discretization methods."""

import logging
from collections.abc import Callable
from math import lgamma, log

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def optimize_bins(  # noqa: PLR0913 # Useful optimization parameters and internal function
    data: npt.NDArray[np.float64],
    cost_function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
    minimize: bool = True,
    window_size: int = 20,
    trend_patience: int = 10,
    stationary_threshold: float = 1e-3,
    method: str = "unknown",
) -> npt.NDArray[np.float64]:
    """
    Find optimal binning by scanning until cost function stops improving.

    Stop early when the criterion gets worse or becomes stationary.

    This function optimizes the number of bins by evaluating a cost function.
    It uses a moving average approach to detect when the cost function either:
    1. Becomes stationary (changes less than stationary_threshold)
    2. Shows a consistent worsening trend

    The optimization stops early if either condition persists for trend_patience
    consecutive windows. The counters reset whenever a new optimum is found.

    Args:
    ----
        data: Input array to be binned
        cost_function: Function that computes cost for given histogram and bins
        minimize: If True, look for minimum. If False, look for maximum
        window_size: Size of window for moving average calculation
        trend_patience: Number of consecutive worse/stationary windows before stopping
        stationary_threshold: Maximum relative change threshold to detect stationarity
        method: Name of binning method being used (for logging)

    Returns:
    -------
        npt.NDArray[np.float64]: Optimal bin edges for the input data

    """
    n = len(data)
    min_edges = 3
    max_edges = min(n, int(np.sqrt(n) * 10)) + 1

    costs = []
    best_cost = float("inf") if minimize else float("-inf")
    best_n = min_edges

    # Walk initial window size for moving average calculation
    for n_bins in range(min_edges, min_edges + window_size):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        hist, _ = np.histogram(data, bins)
        cost = cost_function(hist, bins)
        costs.append(cost)

        if (minimize and cost < best_cost) or (not minimize and cost > best_cost):
            best_cost = cost
            best_n = n_bins

    worse_trend_count = 0
    stationary_count = 0
    last_avg = np.mean(costs[-window_size:])

    for n_bins in range(min_edges + window_size, max_edges):
        bins = np.linspace(np.min(data), np.max(data), n_bins)
        hist, _ = np.histogram(data, bins)
        cost = cost_function(hist, bins)
        costs.append(cost)

        if (minimize and cost < best_cost) or (not minimize and cost > best_cost):
            best_cost = cost
            best_n = n_bins

            worse_trend_count = 0
            stationary_count = 0

        if len(costs) >= window_size:
            current_avg = np.mean(costs[-window_size:])
            rel_change = (
                abs((current_avg - last_avg) / abs(last_avg))
                if last_avg != 0
                else last_avg
            )
            is_stationary = rel_change < stationary_threshold
            stationary_count = (stationary_count + 1) if is_stationary else 0

            if minimize:
                relative_to_best = current_avg / (best_cost + 10e-10)
            else:
                relative_to_best = best_cost / (current_avg + 10e-10)

            far_from_peak = relative_to_best > 1 + 0.05
            if far_from_peak:
                worse_trend_count += 1
            else:
                worse_trend_count = 0

            if worse_trend_count > trend_patience or stationary_count > trend_patience:
                break
            last_avg = current_avg
    else:
        if max_edges < n:
            logger.warning(
                f"Warning: exited bin optimization after {max_edges} bins without "
                f"identifying optimum. Method: {method}."
            )

    logger.info(f"optimization yield: {best_n} bin edges")
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
        return float(
            aic + correction - 2 * m
        )  # Subtract unscaled model dimension term and add corrected.
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
    return optimize_bins(data, knuth_cost, minimize=False, method="Knuth")


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
    return optimize_bins(data, shimazaki_cost, minimize=True, method="Shimazaki")


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
    return optimize_bins(data, aic_cost, minimize=True, method="AIC")


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
    return optimize_bins(data, bic_cost, minimize=True, method="BIC")


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
    return optimize_bins(data, aicc_cost, minimize=True, method="Small Sample AIC")


def cv_cost(hist: npt.NDArray[np.int64], bins: npt.NDArray[np.float64]) -> float:
    """
    Calculate the cross-validation cost function.

    Implementation of cross-validation estimator for histogram.
    Based on equation: J(h) = 2/(h(n+1)) * sum(p_j^2)/(h(n-1))

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

    p = hist / n

    cost = (2 - (n + 1) * sum(p * p)) / (h * (n - 1))
    return float(cost)


def cv_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Find optimal bins using cross-validation method.

    Args:
    ----
        data: Input data array

    Returns:
    -------
        Array of optimal bin edges

    """
    return optimize_bins(data, cv_cost, minimize=True, method="Cross-Validation")
