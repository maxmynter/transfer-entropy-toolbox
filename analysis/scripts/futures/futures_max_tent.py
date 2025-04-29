"""Calculate the TE with the Maximum Entropy Criterion."""

from typing import TypeVar

import numpy as np
import polars as pl
from futures_constants import (
    DATA_PATH,
    MIN_TICKS_PER_DAY,
    RETURNS_DATA,
    TentCalcConfig,
)
from util import (
    build_rolling_pairwise_measure_df,
    date_return_cols,
    generate_filename,
    get_transfer_entropy_surros_for_bins,
    prepare_data,
    return_cols,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols

from te_toolbox.binning.entropy_maximising import max_tent

T = TypeVar("T", bound=InstrumentCols)
TREND_PATIENCE = 50  # Make longer to avoid early stopping for fixed range
config = TentCalcConfig()


def get_maximised_te_fixed_range(  # noqa: PLR0913
    src: T,
    tgt: T,
    df: pl.DataFrame,
    lowest: float,
    highest: float,
    window_size: int = 20,
    trend_patience: int = 10,
) -> tuple[np.float64, np.float64, np.float64]:
    """Get TE for bins over fixed range with maximum transfer entropy."""
    data, at = prepare_data(src, tgt, df)
    bins = max_tent(
        config.TE,
        data,
        lag=config.LAG,
        at=at,
        lower_bound=lowest,
        upper_bound=highest,
        window_size=window_size,
        trend_patience=trend_patience,
    )
    return get_transfer_entropy_surros_for_bins(src, tgt, df, bins)


TE_CALC_FN = get_maximised_te_fixed_range
filename = generate_filename(config, TE_CALC_FN)
if __name__ == "__main__":
    analysis_cols = Cols.get_all_instruments()

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

    lowest, highest = df[return_cols].to_numpy().min(), df[return_cols].to_numpy().max()
    tents = build_rolling_pairwise_measure_df(
        df,
        analysis_cols,
        lambda src, tgt, df: TE_CALC_FN(
            src, tgt, df, lowest=lowest, highest=highest, trend_patience=TREND_PATIENCE
        ),
    )

    print("=== TENTS === ")
    with pl.Config(set_tbl_rows=29):
        print(tents)
    print(tents.describe())
    tents.write_csv(DATA_PATH / f"{filename}.csv")
