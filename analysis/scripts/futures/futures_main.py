"""Analyse the futures time series during COVID."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from futures_constants import (
    MIN_TICKS_PER_DAY,
    PLOT_PATH,
    RETURNS_DATA,
    TE_DATA_PATH,
    TentCalcConfig,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols, TEColumns

from te_toolbox.binning.entropy_maximising import max_tent, max_tent_bootstrap
from te_toolbox.preprocessing import ft_surrogatization
from te_toolbox.stats import autocorrelation

config = TentCalcConfig()

return_cols = [
    Cols.VG.log_returns_5m,
    Cols.ES.log_returns_5m,
    Cols.HS.log_returns_5m,
    Cols.NK.log_returns_5m,
    Cols.CO.log_returns_5m,
]
date_return_cols = [Cols.Date, *return_cols]


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
    source = df[src.get_returns(config.on_column)]
    target = df[tgt.get_returns(config.on_column)]

    data = np.column_stack([target, source])
    at = (0, 1)  # TE source -> target for [target, source]
    return data, at


def linear_te_mean(
    data: npt.NDArray, bins: npt.NDArray, at: tuple[int, int]
) -> np.float64:
    """Calculate TE for only linear correlations with FT surrogates."""
    tol = 10e-10

    min_val = bins[0]
    max_val = bins[-1]
    linears = np.zeros(config.n_bootstrap)

    for i in range(config.n_bootstrap):
        lin_data = data.copy()
        surr_data = ft_surrogatization(lin_data[:, at], config.rng)

        # Rescale so we can use the same bins as the original
        for j in at:
            numerator = (surr_data[:, j] - surr_data[:, j].min()) * (max_val - min_val)
            denum = surr_data[:, j].max() - surr_data[:, j].min()
            surr_data[:, j] = numerator / denum + min_val
            if (
                surr_data[:, j].min() < min_val - tol
                or surr_data[:, j].max() > max_val + tol
            ):
                raise ValueError(f"FT rescaling failed failed beyond tolerance {tol}")
            else:
                surr_data[:, j] = np.clip(surr_data[:, j], min_val, max_val)

        lin_data[:, at] = surr_data
        linears[i] = config.TE(lin_data, bins, config.LAG, at)

    return np.mean(linears)


def bootstrapped_te_mean(
    data: npt.NDArray, bins: npt.NDArray, at: tuple[int, int]
) -> np.float64:
    """Calculate the mean of TE of permuted values."""
    boot_te = np.zeros(config.n_bootstrap)
    for i in range(config.n_bootstrap):
        bs_data = data.copy()
        bs_data[:, 0] = config.rng.permutation(data[:, 0])
        bs_data[:, 1] = config.rng.permutation(data[:, 1])
        boot_te[i] = config.TE(bs_data, bins, config.LAG, at)
    return np.mean(boot_te, dtype=np.float64)


def get_transfer_entropy_for_bins(
    src: T,
    tgt: T,
    df: pl.DataFrame,
    bins: npt.NDArray,
) -> tuple[np.float64, np.float64 | None]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)
    te = np.float64(config.TE(data, bins, config.LAG, at))
    sig_te = np.max([0, te - bootstrapped_te_mean(data, bins, at)])
    if config.get_nonlinear:
        linears = linear_te_mean(data, bins, at)
        return sig_te, np.max(0, sig_te - linears)
    else:
        return sig_te, None


def get_quantil_binned_te(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[np.float64, np.float64 | None]:
    """Calculate TE for fixed with quantil bins.

    (added for consistency checks with 2021 M.Sc. thesis consistency).
    """
    data, at = prepare_data(src, tgt, df)
    bins = np.array(
        [data.min(), np.quantile(data, 0.05), 0.0, np.quantile(data, 0.95), data.max()]
    )

    return get_transfer_entropy_for_bins(src, tgt, df, bins)


def get_maximised_te(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[np.float64, np.float64 | None]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)

    bins = max_tent(config.TE, data, lag=config.LAG, at=at)
    return get_transfer_entropy_for_bins(src, tgt, df, bins)


def get_bootstrap_maximised_te(
    src: T, tgt: T, df: pl.DataFrame
) -> tuple[np.float64, np.float64 | None]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)

    bins = max_tent_bootstrap(config.TE, data, lag=config.LAG, at=at)
    return get_transfer_entropy_for_bins(src, tgt, df, bins)


def create_acf_df(acf_values, max_lag=None):
    """Create polars DataFrame with autocorrelation."""
    if max_lag is None:
        max_lag = len(acf_values)

    return pl.DataFrame({"lag": np.arange(max_lag), "value": acf_values[:max_lag]})


def build_rolling_pairwise_measure_df(
    df: pl.DataFrame,
    cols: list[InstrumentCols],
    measure: Callable[[InstrumentCols, InstrumentCols, pl.DataFrame], np.float64],
    window_days: int = 7,
) -> pl.DataFrame:
    """Build a rolling window TE dataframe with daily steps.

    Args:
        df: Input dataframe with datetime index
        cols: List of instruments to analyze
        measure: Transfer entropy measure function
        window_days: Size of rolling window in days (default 7)

    """
    filter_alias = "filter_var"

    all_dates = (
        df.select(pl.col(Cols.Date).dt.date().alias(filter_alias))
        .unique()
        .sort(filter_alias)
    )

    pairs = [(src, tgt) for src in cols for tgt in cols if src != tgt]
    te_timeseries: list[dict[str, np.float64]] = []

    for current_date in all_dates[filter_alias]:
        print(f"Processing: {current_date}")

        # Calculate window bounds
        window_end = current_date
        window_start = current_date - pl.duration(days=window_days - 1)

        # Get data for current window
        window_df = df.filter(
            (pl.col(Cols.Date).dt.date() <= window_end)
            & (pl.col(Cols.Date).dt.date() >= window_start)
        )

        # Calculate TE for all pairs in this window
        timestep_values = {
            TEColumns.get_te_column_name(src, tgt): measure(src, tgt, window_df)
            for src, tgt in pairs
        }
        timestep_values[Cols.Date] = current_date
        te_timeseries.append(timestep_values)

    return pl.DataFrame(te_timeseries)


def plot_acf(acf_df):
    """Plot the autocorrelations."""
    plt.figure(figsize=(12, 6))
    plt.plot(acf_df["lag"], acf_df["value"], "b-", label="ACF")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_PATH / "autocorrelation.png")


if __name__ == "__main__":
    rng = config.rng
    futures = FuturesDataBuilder.load(RETURNS_DATA)

    df_builder = (
        futures.with_datetime_index()
        .drop_nulls()
        # .slice_before(FuturesDataInfo.no_missing_start.value + timedelta(days=3))
        .log_returns()
        .drop_incomplete_trading_days(MIN_TICKS_PER_DAY)
    )

    df = df_builder.build().select(date_return_cols)

    print(df.head())
    print(df.describe())

    # Plot the returns for crosschecking
    plot_ts(df, return_cols)

    autocorr = create_acf_df(
        autocorrelation(
            df[Cols.CO.log_returns_5m], df[Cols.ES.log_returns_5m], max_lag=25
        )
    )
    plot_acf(autocorr)

    analysis_cols = [Cols.CO, Cols.VG, Cols.ES]

    tents = build_rolling_pairwise_measure_df(
        df_builder.remap_gaussian(
            cols=analysis_cols, source_col=InstrumentCols.log_returns_5m, rng=rng
        ).build(),
        analysis_cols,
        lambda src, tgt, df: get_bootstrap_maximised_te(src, tgt, df),
    )

    plot_ts(
        tents,
        TEColumns.get_pairwise_te_column_names(analysis_cols),
        filename="tents_ts.png",
    )

    print("=== TENTS === ")
    with pl.Config(set_tbl_rows=29):
        print(tents)
    print(tents.describe())
    tents.write_csv(TE_DATA_PATH)
