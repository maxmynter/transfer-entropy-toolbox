"""Single Source of truth for columns of the Futures Dataframe."""

from dataclasses import dataclass, field
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
class ColsAccessor:
    """Futures data column accessor."""

    Date = "Date"
    _instruments: dict[str, InstrumentCols] = field(default_factory=dict)

    def __init__(self):
        """Initialilze all Instrument columns from Instruments Enum."""
        for inst in Instruments:
            self._instruments[inst.name] = InstrumentCols(inst.value)

    def __getattr__(self, instrument_name):
        """Get the instrument columns."""
        if instrument_name in self._instruments:
            return self._instruments[instrument_name]
        raise AttributeError(f"ColsAccessor has no attribute {instrument_name}")


Cols = ColsAccessor()
