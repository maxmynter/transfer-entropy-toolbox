"""Dataset builder for the futures data frame."""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from te_toolbox.preprocessing import remap_to

from .columns import Cols, InstrumentCols, Instruments


class FuturesDataBuilder:
    """Polars DataFrame wrapper that contains wrangling utilities."""

    def __init__(self, df: pl.DataFrame):
        """Wrap data frame in builder."""
        self.df = df

    @classmethod
    def load(cls, path: Path) -> "FuturesDataBuilder":
        """Load csv into builder."""
        return FuturesDataBuilder(pl.read_csv(path, has_header=True))

    def with_datetime_index(self) -> "FuturesDataBuilder":
        """Coerce the date column to a datetime index."""
        df = self.df.with_columns(
            pl.col(Cols.Date).str.strptime(pl.Date, format="%d.%m.%y %H:%M")
        )
        return FuturesDataBuilder(df)

    def drop_cols(self, columns: list[str]) -> "FuturesDataBuilder":
        """Drop columns in argument list."""
        return FuturesDataBuilder(self.df.drop(columns))

    def drop_nans(self) -> "FuturesDataBuilder":
        """Drop NaN rows."""
        return FuturesDataBuilder(self.df.drop_nans())

    def drop_nulls(self) -> "FuturesDataBuilder":
        """Drop null rows."""
        return FuturesDataBuilder(self.df.drop_nulls())

    def log_returns(self) -> "FuturesDataBuilder":
        """Create log returns column."""
        df = self.df
        for instrument in Instruments:
            inst = getattr(Cols, instrument.name)

            df = df.with_columns(
                pl.col(inst.returns).add(1).log().alias(inst.log_returns_5m)
            )
        return FuturesDataBuilder(df)

    def drop_incomplete_trading_days(self, min_bars: int) -> "FuturesDataBuilder":
        """Drop all rows from dates taht have fewer than min_bars observations.

        Args:
        ----
            min_bars: minimum number of rows per date (274 is a full trading day)

        """
        count_col = "count"
        valid_dates = (
            self.df.group_by(pl.col(Cols.Date).dt.date())
            .agg(pl.len().alias(count_col))
            .filter(pl.col(count_col) >= min_bars)
            .get_column(Cols.Date)
        )

        return FuturesDataBuilder(
            self.df.filter(pl.col(Cols.Date).dt.date().is_in(valid_dates))
        )

    def slice_after(self, timestamp: datetime) -> "FuturesDataBuilder":
        """Take all rows after timestamp (inclusive)."""
        df = self.df.filter(pl.col(Cols.Date) >= timestamp)
        return FuturesDataBuilder(df)

    def slice_before(self, timestamp: datetime) -> "FuturesDataBuilder":
        """Take all rows before timestamp (inclusive)."""
        df = self.df.filter(pl.col(Cols.Date) <= timestamp)
        return FuturesDataBuilder(df)

    def remap_uniform(
        self,
        cols: list[InstrumentCols],
        source_col=InstrumentCols.returns_5m,
        rng: np.random.Generator | None = None,
    ) -> "FuturesDataBuilder":
        """Remap the data rank ordered to the uniform [0,1] interval."""
        df = self.df
        for col in cols:
            source_name = source_col.__get__(col)
            target_name = col.unif_remap_returns
            df = df.with_columns(
                pl.Series(
                    target_name,
                    remap_to(
                        df[source_name].to_numpy(),
                        np.random.uniform(low=0, high=1, size=len(df[source_name])),
                        rng,
                    ),
                )
            )
        return FuturesDataBuilder(df)

    def build(self) -> pl.DataFrame:
        """Unwrap the dataframe from builder."""
        return self.df
