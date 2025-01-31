"""Analyse the futures time series during COVID."""

from futures_constants import DATA_PATH
from wrangling import Cols, FuturesDataBuilder

if __name__ == "__main__":
    futures = FuturesDataBuilder.load(DATA_PATH)
    print(
        futures.with_datetime_index()
        .drop_ticks()
        .log_returns()
        .drop_nulls()
        .build()
        .select([Cols.Date, Cols.VG.close, Cols.VG.returns])
        .head(n=15)
    )
    df = (
        futures.with_datetime_index()
        .drop_nulls()
        .log_returns()
        .volatility()
        .volume_weighted_close()
        .build()
    )

    print(
        df.select(
            [Cols.Date, Cols.NK.close, Cols.NK.returns, Cols.NK.volatility]
        ).head()
    )
