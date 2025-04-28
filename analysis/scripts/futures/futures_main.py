"""Analyse the futures time series during COVID."""

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
    create_acf_df,
    date_return_cols,
    generate_filename,
    get_transfer_entropy_surros_for_bins,
    plot_acf,
    plot_ts,
    prepare_data,
    return_cols,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols, TEColumns
from wrangling.columns import ReturnType

from te_toolbox.binning.entropy_maximising import max_tent_bootstrap
from te_toolbox.stats import autocorrelation

config = TentCalcConfig()

T = TypeVar("T", bound=InstrumentCols)


def get_bootstrap_maximised_te(
    src: T, tgt: T, df: pl.DataFrame, window_size: int = 20, trend_patience: int = 10
) -> tuple[np.float64, np.float64, np.float64]:
    """Calculate TE between variables for dataset."""
    data, at = prepare_data(src, tgt, df)

    bins = max_tent_bootstrap(
        config.TE,
        data,
        lag=config.LAG,
        at=at,
        window_size=window_size,
        trend_patience=trend_patience,
    )
    return get_transfer_entropy_surros_for_bins(src, tgt, df, bins)


TE_CALC_FN = get_bootstrap_maximised_te
filename = generate_filename(config, TE_CALC_FN)
if __name__ == "__main__":
    analysis_cols = Cols.get_all_instruments()  # [Cols.CO, Cols.VG, Cols.ES]

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

    if config.on_column == ReturnType.UNIFORM:
        remapped_df = df_builder.remap_uniform(
            cols=analysis_cols, source_col=InstrumentCols.log_returns_5m, rng=rng
        ).build()
    else:
        remapped_df = df_builder.build()

    print(remapped_df.describe())

    tents = build_rolling_pairwise_measure_df(
        remapped_df,
        analysis_cols,
        lambda src, tgt, df: TE_CALC_FN(src, tgt, df),
    )

    plot_ts(
        tents,
        TEColumns.get_pairwise_te_column_names(analysis_cols),
        filename=(filename + ".png"),
    )

    print("=== TENTS === ")
    with pl.Config(set_tbl_rows=29):
        print(tents)
    print(tents.describe())
    tents.write_csv(DATA_PATH / f"{filename}.csv")
