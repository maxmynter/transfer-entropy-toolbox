"""Autocorrelation function."""

import numpy as np
import numpy.typing as npt


def autocorrelation(x: npt.NDArray, y: npt.NDArray, max_lag: int | None = None):
    """
    Calculate the autocorrelation function between two series.

    Parameters
    ----------
    x : numpy.ndarray
        First time series
    y : numpy.ndarray, optional
        Second time series. If None, calculates autocorrelation of x with itself
    max_lag : int, optional
        Maximum lag to calculate. If None, calculates for all possible lags

    Returns
    -------
    numpy.ndarray
        Array of autocorrelation values for each lag

    """
    if y is None:
        y = x

    x = np.asarray(x)
    y = np.asarray(y)

    if max_lag is None:
        max_lag = len(x)

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    norm = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        corr = np.sum(x_centered[lag:] * y_centered[: -lag if lag > 0 else None])
        acf[lag] = corr / norm

    return acf
