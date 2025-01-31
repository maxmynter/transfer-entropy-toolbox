"""Single Source of truth for columns of the Futures Dataframe."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class FuturesColumnGroup:
    """Group of Variables relating to a single Future."""

    base: str
    close: str
    volume: str
    number: str
    ticks: str

    @property
    def returns(self) -> str:
        """Get returns column name."""
        return f"{self.base}_returns"

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
            self.close,
            self.volume,
            self.number,
            self.ticks,
            self.returns,
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
            close=f"{ticker}_Close",
            volume=f"{ticker}_Volume",
            number=f"{ticker}_Number",
            ticks=f"{ticker}_Ticks",
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
                    instrument.close,
                    instrument.volume,
                    instrument.number,
                    instrument.ticks,
                ]
            )
        return columns


Cols = FuturesColumns()  # Column Singleton
