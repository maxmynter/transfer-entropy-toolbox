"""Single Source of truth for columns of the Futures Dataframe."""

from dataclasses import dataclass
from enum import Enum


class Instruments(Enum):
    """Base names of the futures instruments."""

    ES = "ES1"  # E-mini S&P 500
    VG = "VG1"  # Euro Stoxx 50
    HS = "HI1"  # Hang Seng
    NK = "NK1"  # Nikkei 225
    CO = "CO1"  # Crude Oil

    def __str__(self) -> str:
        """Return string value in string context."""
        return self.value


@dataclass(frozen=True)
class InstrumentCols:
    """Holds columns corresponding to an instrument."""

    base: str

    @property
    def returns(self) -> str:
        """Return the returns column name."""
        return f"{self.base} returns"

    @property
    def log_returns_5m(self) -> str:
        """Return the log returns column name."""
        return f"{self.base} log returns (5m)"


@dataclass
class Columns:
    """Futures data column accessor."""

    ES = InstrumentCols(Instruments.ES.value)
    VG = InstrumentCols(Instruments.VG.value)
    HS = InstrumentCols(Instruments.HS.value)
    NK = InstrumentCols(Instruments.NK.value)
    CO = InstrumentCols(Instruments.CO.value)
    Date = "Date"

    def __init__(self):
        """Initialilze all Instrument columns from Instruments Enum."""
        for inst in Instruments:
            setattr(self, inst.name, InstrumentCols(inst.value))


Cols = Columns()
