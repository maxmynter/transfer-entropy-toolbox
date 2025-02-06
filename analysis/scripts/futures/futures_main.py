"""Analyse the futures time series during COVID."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from futures_constants import (
    DATA_PATH,
    MIN_TICKS_PER_DAY,
    PLOT_PATH,
    TentCalcConfig,
    TimeGranularity,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols, TEColumns

from te_toolbox.binning.entropy_maximising import max_tent
from te_toolbox.stats import autocorrelation

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
        ax.plot(df[col], label=col, alpha=0.6)
    ax.legend()
    plt.savefig(PLOT_PATH / filename)
    plt.close()


def get_maximised_transfer_entropy(
    src: InstrumentCols, tgt: InstrumentCols, df: pl.DataFrame, config: TentCalcConfig
) -> np.float64:
    """Calculate TE between variables for dataset."""
    source = df[src.get_returns(config.on_column)]
    target = df[tgt.get_returns(config.on_column)]

    data = np.column_stack([target, source])
    at = (0, 1)  # TE source -> target for [target, source]
    bins = max_tent(config.TE, data, lag=TentCalcConfig().LAG, at=at)
    return np.float64(config.TE(data, bins, TentCalcConfig().LAG, at))


def get_normalized_quantil_binned_te(
    src: InstrumentCols, tgt: InstrumentCols, df: pl.DataFrame, config: TentCalcConfig
) -> np.float64:
    """Calculate TE for fixed with quantil bins.

    (added for consistency checks with 2021 M.Sc. thesis consistency).
    """
    source = df[src.get_returns(config.on_column)]
    target = df[tgt.get_returns(config.on_column)]

    data = np.column_stack([target, source])

    bins = np.array(
        [data.min(), np.quantile(data, 0.05), 0.0, np.quantile(data, 0.95), data.max()]
    )

    return np.float64(config.TE(data, bins, TentCalcConfig().LAG, (0, 1)))


def get_transfer_entropy_for_bins(
    src: InstrumentCols,
    tgt: InstrumentCols,
    df: pl.DataFrame,
    bins: npt.NDArray,
    config: TentCalcConfig,
) -> np.float64:
    """Calculate TE between variables for dataset."""
    source = df[src.get_returns(config.on_column)]
    target = df[tgt.get_returns(config.on_column)]

    data = np.column_stack([target, source])
    at = (0, 1)  # TE source -> target for [target, source]
    return np.float64(config.TE(data, bins, TentCalcConfig().LAG, at))


def build_pairwise_measure_df(
    df: pl.DataFrame,
    cols: list[InstrumentCols],
    measure: Callable[[InstrumentCols, InstrumentCols, pl.DataFrame], np.float64],
    config: TentCalcConfig,
) -> pl.DataFrame:
    """Build a per day TE dataframe from df."""
    filter_alias = "filter_var"

    timesteps = (
        df.select(
            pl.col(Cols.Date).dt.truncate(config.granularity.value).alias(filter_alias)
            if config.granularity != TimeGranularity.DAY
            else df.select(pl.col(Cols.Date).alias(filter_alias))
        )
        .unique()
        .sort(filter_alias)
    )

    pairs = [(src, tgt) for src in cols for tgt in cols if src != tgt]

    te_timeseries: list[dict[str, np.float64]] = []

    for timestep in timesteps[filter_alias]:
        print(f"Processing: {timestep}")

        timestep_df = df.filter(
            pl.col(Cols.Date).dt.truncate(config.granularity.value) == timestep
            if config.granularity != TimeGranularity.DAY
            else pl.col(Cols.Date) == timestep
        )

        timestep_values = {
            TEColumns.get_te_column_name(src, tgt): measure(src, tgt, timestep_df)
            for src, tgt in pairs
        }
        timestep_values[Cols.Date] = timestep
        te_timeseries.append(timestep_values)

    return pl.DataFrame(te_timeseries)


def create_acf_df(acf_values, max_lag=None):
    """Create polars DataFrame with autocorrelation."""
    if max_lag is None:
        max_lag = len(acf_values)

    return pl.DataFrame({"lag": np.arange(max_lag), "value": acf_values[:max_lag]})


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
    rng = np.random.default_rng(42)
    futures = FuturesDataBuilder.load(DATA_PATH)

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

    analysis_cols = [Cols.CO, Cols.ES]
    tents = build_pairwise_measure_df(
        df_builder.remap_uniform(
            cols=analysis_cols, source_col=InstrumentCols.log_returns_5m, rng=rng
        ).build(),
        analysis_cols,
        lambda src, tgt, df: get_transfer_entropy_for_bins(
            src, tgt, df, np.array([0, 0.5, 1]), TentCalcConfig()
        ),
        TentCalcConfig(),
    )

    plot_ts(
        tents,
        TEColumns.get_pairwise_te_column_names([Cols.CO, Cols.ES]),
        filename="tents_ts.png",
    )

    print("=== TENTS === ")
    with pl.Config(set_tbl_rows=29):
        print(tents)
    print(tents.describe())
