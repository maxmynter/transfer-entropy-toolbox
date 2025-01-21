"""Rule based discretization methods."""

import numpy as np
import numpy.typing as npt


def sqrt_n_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using square root of n rule.

    Args:
    ----
        data: Input array of values

    Returns:
    -------
        Array of bin edges

    """
    n_bins = int(np.ceil(np.sqrt(len(data))))
    bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
    return bins


def sturges_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using Sturges' formula: 1 + log2(n).

    Args:
    ----
        data: Input array of values

    Returns:
    -------
        Array of bin edges

    """
    n_bins = int(np.ceil(np.log2(len(data))) + 1)
    bins = np.linspace(np.min(data), np.max(data), n_bins + 1)

    return bins


def rice_rule_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using Rice's rule: 2 * n^(1/3).

    Args:
    ----
        data: Input array of values

    Returns:
    -------
        Array of bin edges

    """
    n_bins = int(np.ceil(2 * np.cbrt(len(data))))
    bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
    return bins


def doanes_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using Doane's formula, which adjusts for skewness.

    Args:
    ----
        data: Input array of values

    Returns:
    -------
        Array of bin edges

    """
    n = len(data)
    g1 = np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))

    n_bins = int(1 + np.log2(n) + np.log2(1 + abs(g1) / sigma_g1))
    bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
    return bins


def scotts_rule_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using Scott's rule.

    It is optimal for normally distributed data.

    Args:
    ----
        data: Input array of values
        allow_empty: If False, removes empty bins

    Returns:
    -------
        Array of bin edges

    """
    h = 3.49 * np.std(data) / np.cbrt(len(data))
    bins = np.arange(np.min(data), np.max(data) + h, h)
    return bins


def freedman_diaconis_bins(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate bin edges using Freedman-Diaconis rule: h = 2 * IQR(X) / n^(1/3).

    Args:
    ----
        data: Input array of values

    Returns:
    -------
        Array of bin edges

    """
    # Calculate IQR
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    # Calculate bin width
    h = 2 * iqr / np.cbrt(len(data))

    # Create bins from min to max with width h
    bins = np.arange(np.min(data), np.max(data) + h, h)
    return bins
