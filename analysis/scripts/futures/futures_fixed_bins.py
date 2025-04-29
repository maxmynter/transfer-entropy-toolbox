"""Calculate the transfer entropy for fixed binnings."""

import numpy as np
from futures_constants import (
    DATA_PATH,
    MIN_TICKS_PER_DAY,
    RETURNS_DATA,
    TentCalcConfig,
)
from futures_main import (
    build_rolling_pairwise_measure_df,
)
from numpy.typing import NDArray
from util import (
    date_return_cols,
    generate_filename,
    get_transfer_entropy_surros_for_bins,
    return_cols,
)
from wrangling import Cols, FuturesDataBuilder

config = TentCalcConfig()

N_BINS = [2**n for n in range(1, 11)]


if __name__ == "__main__":
    analysis_cols = Cols.get_all_instruments()
    futures = FuturesDataBuilder.load(RETURNS_DATA)

    df_builder = (
        futures.with_datetime_index()
        .drop_nulls()
        .log_returns()
        .drop_incomplete_trading_days(MIN_TICKS_PER_DAY)
    )
    df = df_builder.build().select(date_return_cols)
    print(df.describe())

    lowest, highest = df[return_cols].to_numpy().min(), df[return_cols].to_numpy().max()

    for n_bins in N_BINS:
        print(f"=== {n_bins} BINS ===")
        bins: NDArray[np.float64] = np.linspace(lowest, highest, num=n_bins + 1)
        bins_sym_about_zero: NDArray[np.float64] = np.concatenate(
            [
                np.linspace(lowest, 0, num=n_bins // 2 + 1),
                np.linspace(0, highest, num=n_bins // 2 + 1)[1:],
            ]
        )
        fixed_n_bins_filename = generate_filename(
            config, f"fixed_{n_bins}_bins_over_range"
        )
        bin_tents = build_rolling_pairwise_measure_df(
            df,
            analysis_cols,
            lambda src, tgt, df, bins=bins: get_transfer_entropy_surros_for_bins(
                src, tgt, df, bins
            ),
        )
        bin_tents.write_csv(DATA_PATH / f"{fixed_n_bins_filename}.csv")
        fixed_n_sym_bins_filename = generate_filename(
            config, f"fixed_{n_bins}_bins_sym_abt_0"
        )
        sym_bin_tents = build_rolling_pairwise_measure_df(
            df,
            analysis_cols,
            lambda src,
            tgt,
            df,
            sym_bins=bins_sym_about_zero: get_transfer_entropy_surros_for_bins(
                src, tgt, df, sym_bins
            ),
        )
        sym_bin_tents.write_csv(DATA_PATH / f"{fixed_n_sym_bins_filename}.csv")
