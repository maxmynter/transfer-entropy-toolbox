"""Calculate the transfer entropy for fixed binnings."""

import numpy as np
from futures_constants import (
    DATA_PATH,
    MIN_TICKS_PER_DAY,
    PLOT_PATH,
    RETURNS_DATA,
    TentCalcConfig,
)
from wrangling import Cols, FuturesDataBuilder, InstrumentCols, TEColumns

config = TentCalcConfig()

N_BINS = [2**n for n in range(1, 11)]

return_cols = [
    Cols.VG.log_returns_5m,
    Cols.ES.log_returns_5m,
    Cols.HS.log_returns_5m,
    Cols.NK.log_returns_5m,
    Cols.CO.log_returns_5m,
]
date_return_cols = [Cols.Date, *return_cols]

if __name__ == "__main__":
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
        bins = np.linspace(lowest, highest, num=n_bins + 1)
        bins_sym_about_zero = np.concatenate(
            [
                np.linspace(lowest, 0, num=n_bins // 2 + 1),
                np.linspace(0, highest, num=n_bins // 2 + 1)[1:],
            ]
        )
