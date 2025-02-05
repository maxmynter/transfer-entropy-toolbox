"""Analyse the futures time series during COVID."""

import numpy as np
import polars as pl
from futures_constants import (
    DATA_PATH,
    LAG,
    MIN_TICKS_PER_DAY,
    TE,
    FuturesDataInfo,
    TimeGranularity,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols

from te_toolbox.binning.entropy_maximising import max_tent

date_return_cols = [
    Cols.Date,
    Cols.VG.log_returns_5m,
    Cols.ES.log_returns_5m,
    Cols.HS.log_returns_5m,
    Cols.NK.log_returns_5m,
    Cols.CO.log_returns_5m,
]


def get_transfer_entropy(
    src: InstrumentCols,
    tgt: InstrumentCols,
    df: pl.DataFrame,
    tent: TE = TE.LOGN,
) -> np.float64:
    """Calculate TE between variables for dataset."""
    source = df[src.log_returns_5m]
    target = df[src.log_returns_5m]

    data = np.column_stack([target, source])
    at = (0, 1)  # TE source -> target for [target, source]
    bins = max_tent(tent, data, lag=LAG, at=at)
    return np.float64(tent(data, bins, LAG, at))


def build_te_df(
    df: pl.DataFrame,
    cols=list[InstrumentCols],
    granularity: TimeGranularity = TimeGranularity.DAY,
    tent: TE = TE.LOGN,
) -> pl.DataFrame:
    """Build a per day TE dataframe from df."""
    filter_alias = "filter_var"

    timesteps = df.select(
        pl.col(Cols.Date).dt.truncate(granularity.value).alias(filter_alias)
        if granularity != TimeGranularity.DAY
        else df.select(pl.col(Cols.Date).alias(filter_alias))
    ).unique()

    pairs = [(src, tgt) for src in cols for tgt in cols if src != tgt]

    te_timeseries: list[dict[str, np.float64]] = []

    for timestep in timesteps[filter_alias]:
        print(f"Processing: {timestep}")

        timestep_df = df.filter(
            pl.col(Cols.Date).dt.truncate(granularity.value) == timestep
            if granularity != TimeGranularity.DAY
            else pl.col(Cols.Date) == timestep
        )

        timestep_values = {
            f"{src.base}->{tgt.base}": get_transfer_entropy(src, tgt, timestep_df, tent)
            for src, tgt in pairs
        }
        timestep_values[Cols.Date] = timestep
        te_timeseries.append(timestep_values)

    return pl.DataFrame(te_timeseries)


if __name__ == "__main__":
    futures = FuturesDataBuilder.load(DATA_PATH)

    df = (
        futures.with_datetime_index()
        .drop_nulls()
        # .slice_before(FuturesDataInfo.no_missing_start.value + timedelta(days=3))
        .slice_after(FuturesDataInfo.no_missing_start.value)
        .slice_before(FuturesDataInfo.no_missing_end.value)
        .log_returns()
        .drop_incomplete_trading_days(MIN_TICKS_PER_DAY)
        .build()
        .select(date_return_cols)
    )

    print(df.head())
    print(df.describe())

    # TODO: Use autocorrelation function to determine lag
    te_test = TE.TENT(
        np.column_stack([df[Cols.CO.log_returns_5m, Cols.ES.log_returns_5m]]),
        np.array([-10, 0, 10]),
        LAG,
    )
    print(te_test)

    # tents = build_te_df(df, [Cols.CO, Cols.ES], TimeGranularity.WEEK, tent=TE.LOGN)
    # print("=== TENTS === ")
    # print(tents.head())
    # print(tents.describe())
