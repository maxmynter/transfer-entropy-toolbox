"""Constants for the Futures Time Series Analysis."""

from datetime import datetime
from enum import Enum
from pathlib import Path

from te_toolbox.entropies.transfer import (
    logn_normalized_transfer_entropy,
    transfer_entropy,
)


class TE(Enum):
    """Possible TE values for calculation."""

    TENT = transfer_entropy
    LOGN = logn_normalized_transfer_entropy


class TimeGranularity(Enum):
    """Select time granularity on which the TE is calculated."""

    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"


class FuturesDataInfo(Enum):
    """Specific attributes of the futures dataset."""

    no_missing_start: datetime = datetime(2019, 11, 1, 1, 20)
    no_missing_end: datetime = datetime(2020, 5, 15, 11, 25)


LAG = 1

MIN_TICKS_PER_DAY = 200  # Minimum data in a trading day for robust entropy calculation.

DATA_PATH = Path("analysis/data/full_futures/futures_returns.csv")
