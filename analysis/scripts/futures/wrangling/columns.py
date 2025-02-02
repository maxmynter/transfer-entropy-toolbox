"""Single Source of truth for columns of the Futures Dataframe."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class FuturesColumnGroup:
    """Group of Variables relating to a single Future."""

    base: str
    close_returns_5m: str
    volume_5m: str
    number_5m: str
    ticks_5m: str

    @property
    def log_returns_5m(self) -> str:
        """Get log returns column name."""
        return f"{self.base}_log_returns_5m"

    @property
    def volatility(self) -> str:
        """Get volatility column name."""
        return f"{self.base}_vol"

    @property
    def volume_weighted(self) -> str:
        """Get volume_weighted column name."""
        return f"{self.base}_vw"

    def all(self) -> list[str]:
        """Get all column names."""
        return [
            self.close_returns_5m,
            self.volume_5m,
            self.number_5m,
            self.ticks_5m,
            self.log_returns_5m,
            self.volatility,
            self.volume_weighted,
        ]


class FuturesColumns:
    """Futures Column names."""

    class Base(str, Enum):
        """Instruments in raw data."""

        ES = "ES1"
        VG = "VG1"
        HS = "H1"
        NK = "NK1"
        CO = "CO1"

        def __str__(self) -> str:
            """Return string value in string context."""
            return self.value

    Date = "Dates"

    @classmethod
    def _create_column_group(cls, ticker: Base) -> FuturesColumnGroup:
        """Create a column group from ticker symbol."""
        return FuturesColumnGroup(
            base=ticker.value,
            close_returns_5m=f"{ticker}_Close",
            volume_5m=f"{ticker}_Volume",
            number_5m=f"{ticker}_Number",
            ticks_5m=f"{ticker}_Ticks",
        )

    def __init__(self):
        """Create column groups for base tickers."""
        self.ES = FuturesColumns._create_column_group(self.Base.ES)  # E-mini S&P 500
        self.VG = FuturesColumns._create_column_group(self.Base.VG)  # Euro Stoxx 50
        self.HS = FuturesColumns._create_column_group(self.Base.HS)  # Hang Seng
        self.NK = FuturesColumns._create_column_group(self.Base.NK)  # Nikkei 225
        self.CO = FuturesColumns._create_column_group(self.Base.CO)  # Crude Oil

    @property
    def all_instruments(self) -> list[FuturesColumnGroup]:
        """Return all futures data groups."""
        return [self.ES, self.VG, self.HS, self.NK, self.CO]

    def all_columns(self, include_derived=True) -> list[str]:
        """Get all columns."""
        columns = [self.Date]
        for instrument in self.all_instruments:
            columns.extend(
                instrument.all()
                if include_derived
                else [
                    instrument.close_returns_5m,
                    instrument.volume_5m,
                    instrument.number_5m,
                    instrument.ticks_5m,
                ]
            )
        return columns


Cols = FuturesColumns()  # Column Singleton
