"""Shared Utils for Futures Calculations and Plots."""

from collections.abc import Callable, Mapping
from datetime import date
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from futures_constants import PLOT_PATH, TentCalcConfig
from joblib import Parallel, delayed
from wrangling import Cols, InstrumentCols, TEColumns

from te_toolbox.binning.entropy_maximising import max_tent
from te_toolbox.preprocessing import ft_surrogatization

T = TypeVar("T", bound=InstrumentCols)

return_cols = [
    Cols.VG.log_returns_5m,
    Cols.ES.log_returns_5m,
    Cols.HS.log_returns_5m,
    Cols.NK.log_returns_5m,
    Cols.CO.log_returns_5m,
]
date_return_cols = [Cols.Date, *return_cols]


def plot_acf(acf_df):
    """Plot the autocorrelations."""
    plt.figure(figsize=(12, 6))
    plt.plot(acf_df["lag"], acf_df["value"], "b-", label="ACF")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH / "autocorrelation.png")


def generate_filename(config: TentCalcConfig, calc_fn: Callable | str) -> str:
    """Generate the filename from config and calculation function."""
    generating_fn = calc_fn.__name__ if isinstance(calc_fn, Callable) else calc_fn
    filename = (
        f"tents_ts_{config.LAG}Lag_fn={generating_fn}"
        f"_{config.WINDOW_SIZE}window_{config.WINDOW_STEP}step"
        f"sourcecol={config.on_column.value}"
    )
    return filename


def plot_ts(df: pl.DataFrame, cols: list[str], filename="ts.png"):
    """Plot the timeseries."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        ax.plot(df[Cols.Date], df[col], label=col, alpha=0.6)
    ax.legend()
    plt.savefig(PLOT_PATH / filename)
    plt.close()


def prepare_data(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[npt.NDArray, tuple[int, int]]:
    """Prepare the data arrays for TE calculation."""
    source = df[src.get_returns(TentCalcConfig.on_column)]
    target = df[tgt.get_returns(TentCalcConfig.on_column)]

    data = np.column_stack([target, source])
    at = (0, 1)  # TE source -> target for [target, source]
    return data, at


def process_pairwise_step(
    current_date: date,
    df: pl.DataFrame,
    measure: Callable[[T, T, pl.DataFrame], tuple[np.float64, np.float64, np.float64]],
    pairs: list[tuple[T, T]],
) -> Mapping[str, np.float64 | date]:
    """Process transfer entropy of dataframe slice."""
    print(f"Processing: {current_date}")

    window_end = current_date
    window_start = current_date - pl.duration(days=TentCalcConfig.WINDOW_SIZE - 1)

    window_df = df.filter(
        (pl.col(Cols.Date).dt.date() <= window_end)
        & (pl.col(Cols.Date).dt.date() >= window_start)
    )

    timestep_values: dict[str, np.float64 | date] = {}
    for src, tgt in pairs:
        (
            te,
            uncorr_te,
            linear_te,
        ) = measure(src, tgt, window_df)
        timestep_values[TEColumns.get_te_column_name(src, tgt)] = te
        timestep_values[TEColumns.get_bs_column_name(src, tgt)] = uncorr_te
        timestep_values[TEColumns.get_ft_column_name(src, tgt)] = linear_te

    timestep_values[Cols.Date] = current_date
    return timestep_values


def build_rolling_pairwise_measure_df(
    df: pl.DataFrame,
    cols: list[T],
    measure: Callable[[T, T, pl.DataFrame], np.float64],
) -> pl.DataFrame:
    """Build a rolling window TE dataframe with daily steps.

    cols: List of instruments to analyze
    measure: Transfer entropy measure function
    window_days: Size of rolling window in days (default 7)

    """
    all_dates = (
        df.get_column(Cols.Date)
        .unique()
        .sort()
        .to_list()[:: TentCalcConfig.WINDOW_STEP]
    )

    pairs = [(src, tgt) for src in cols for tgt in cols if src != tgt]
    te_timeseries: list[dict[str, np.float64]] = []

    te_timeseries = Parallel(n_jobs=-1)(
        delayed(process_pairwise_step)(date, df, measure, pairs) for date in all_dates
    )

    return pl.DataFrame(te_timeseries)


def bootstrapped_te_mean(
    data: npt.NDArray, bins: npt.NDArray, at: tuple[int, int]
) -> np.float64:
    """Calculate the mean of TE of permuted values."""
    boot_te = np.zeros(TentCalcConfig.n_bootstrap)
    for i in range(TentCalcConfig.n_bootstrap):
        bs_data = data.copy()
        bs_data[:, at[0]] = TentCalcConfig.rng.permutation(data[:, at[0]])
        bs_data[:, at[1]] = TentCalcConfig.rng.permutation(data[:, at[1]])
        boot_te[i] = TentCalcConfig.TE(bs_data, bins, TentCalcConfig.LAG, at)
    return np.mean(boot_te, dtype=np.float64)


def linear_te_mean(
    data: npt.NDArray, bins: npt.NDArray, at: tuple[int, int]
) -> np.float64:
    """Calculate TE for only linear correlations with FT surrogates."""
    tol = 10e-10

    min_val = bins[0]
    max_val = bins[-1]
    linears = np.zeros(TentCalcConfig.n_bootstrap)

    for i in range(TentCalcConfig.n_bootstrap):
        lin_data = data.copy()
        surr_data = ft_surrogatization(lin_data[:, at], TentCalcConfig.rng)

        # Rescale so we can use the same bins as the original
        for j in at:
            numerator = (surr_data[:, j] - surr_data[:, j].min()) * (max_val - min_val)
            denum = surr_data[:, j].max() - surr_data[:, j].min()
            surr_data[:, j] = numerator / (denum + tol) + min_val
            if (
                surr_data[:, j].min() < min_val - tol
                or surr_data[:, j].max() > max_val + tol
            ):
                raise ValueError(f"FT rescaling failed failed beyond tolerance {tol}")
            else:
                surr_data[:, j] = np.clip(surr_data[:, j], min_val, max_val)

        lin_data[:, at] = surr_data
        linears[i] = TentCalcConfig.TE(lin_data, bins, TentCalcConfig.LAG, at)

    return np.mean(linears)


def get_transfer_entropy_surros_for_bins(
    src: T,
    tgt: T,
    df: pl.DataFrame,
    bins: npt.NDArray,
) -> tuple[np.float64, np.float64, np.float64]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)

    te = np.float64(TentCalcConfig.TE(data, bins, TentCalcConfig.LAG, at))

    bootstrap_te = (
        bootstrapped_te_mean(data, bins, at)
        if TentCalcConfig.get_bootstrap
        else np.float64("nan")
    )

    ft_te = (
        linear_te_mean(data, bins, at)
        if TentCalcConfig.get_nonlinear
        else np.float64("nan")
    )

    return te, bootstrap_te, ft_te


def get_quantil_binned_te(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[np.float64, np.float64, np.float64]:
    """Calculate TE for fixed with quantil bins.

    (added for consistency checks with 2021 M.Sc. thesis consistency).
    """
    data, at = prepare_data(src, tgt, df)
    bins = np.array(
        [data.min(), np.quantile(data, 0.05), 0.0, np.quantile(data, 0.95), data.max()]
    )

    return get_transfer_entropy_surros_for_bins(src, tgt, df, bins)


def get_maximised_te(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[np.float64, np.float64, np.float64]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)

    bins = max_tent(TentCalcConfig.TE, data, lag=TentCalcConfig.LAG, at=at)
    return get_transfer_entropy_surros_for_bins(src, tgt, df, bins)


def create_acf_df(acf_values, max_lag=None):
    """Create polars DataFrame with autocorrelation."""
    if max_lag is None:
        max_lag = len(acf_values)

    return pl.DataFrame({"lag": np.arange(max_lag), "value": acf_values[:max_lag]})
