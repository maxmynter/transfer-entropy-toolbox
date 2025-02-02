"""Analyse the futures time series during COVID."""

from futures_constants import DATA_PATH
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
        .log_returns()
        .build()
        .select(date_return_cols)
    )

    print(df.head())
    print(df.tail())
    print(df.describe())
