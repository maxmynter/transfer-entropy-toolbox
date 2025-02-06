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


class ReturnType(Enum):
    """Types of returns available in the dataset."""

    RAW = "returns"
    LOG = "log returns (5m)"
    UNIFORM = "return mapped unif. [-1,1]"


@dataclass(frozen=True)
class InstrumentCols:
    """Holds columns corresponding to an instrument."""

    base: str

    def get_returns(self, ret_type: ReturnType) -> str:
        """Get column name for any return type."""
        return f"{self.base} {ret_type.value}"

    @property
    def returns_5m(self) -> str:
        """Return the returns column name."""
        return self.get_returns(ReturnType.RAW)

    @property
    def log_returns_5m(self) -> str:
        """Return the log returns column name."""
        return self.get_returns(ReturnType.LOG)

    @property
    def unif_remap_returns(self) -> str:
        """Return the column name of the uniform remaps."""
        return self.get_returns(ReturnType.UNIFORM)


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

    def get_all_returns(self, return_type: ReturnType) -> list[str]:
        """Get all return columns of a specific type."""
        return [
            getattr(self, inst.name).get_returns(return_type) for inst in Instruments
        ]


class TEColumns:
    """Contains the TE column names."""

    @classmethod
    def get_te_column_name(cls, src: InstrumentCols, tgt: InstrumentCols) -> str:
        """Get TE from to column name."""
        return f"{src.base}->{tgt.base}"

    @classmethod
    def get_pairwise_te_column_names(
        cls, instruments: list[InstrumentCols]
    ) -> list[str]:
        """Get column names for all TE pairs."""
        return [
            cls.get_te_column_name(src, tgt)
            for src in instruments
            for tgt in instruments
            if src != tgt
        ]


Cols = Columns()
