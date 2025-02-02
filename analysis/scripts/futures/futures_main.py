"""Analyse the futures time series during COVID."""

from futures_constants import DATA_PATH, MIN_TICKS_PER_DAY, FuturesDataInfo
from wrangling import Cols, FuturesDataBuilder

date_return_cols = [
    Cols.Date,
    Cols.VG.log_returns_5m,
    Cols.ES.log_returns_5m,
    Cols.HS.log_returns_5m,
    Cols.NK.log_returns_5m,
    Cols.CO.log_returns_5m,
]

if __name__ == "__main__":
    futures = FuturesDataBuilder.load(DATA_PATH)

    df = (
        futures.with_datetime_index()
        .drop_ticks()
        .drop_nans()
        .drop_nulls()
        .slice_after(FuturesDataInfo.no_missing_start)
        .slice_before(FuturesDataInfo.no_missing_end)
        .log_returns()
        .drop_incomplete_trading_days(MIN_TICKS_PER_DAY)
        .build()
        .select(date_return_cols)
    )

    print(df.head())
    print(df.tail())
    print(df.describe())
