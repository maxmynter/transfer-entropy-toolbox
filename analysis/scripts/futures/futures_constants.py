"""Constants for the Futures Time Series Analysis."""

from datetime import datetime
from enum import Enum
from pathlib import Path


class FuturesDataInfo(Enum):
    """Specific attributes of the futures dataset."""

    no_missing_start: datetime = datetime(2019, 11, 1, 1, 20)
    no_missing_end: datetime = datetime(2020, 5, 15, 11, 25)


DATA_PATH = Path("analysis/data/full_futures/full_futures.csv")
