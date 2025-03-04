"""Constants for the Futures Time Series Analysis."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from wrangling.columns import ReturnType

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


@dataclass
class TentCalcConfig:
    """Configure the TE Calculation."""

    TE: TE = TE.LOGN
    LAG = 1
    WINDOW_SIZE: int = 5
    WINDOW_STEP: int = 1
    n_bootstrap: int = 100
    on_column: ReturnType = ReturnType.LOG
    get_nonlinear: bool = True
    get_bootstrap: bool = True
    rng = np.random.default_rng()


def generate_filename(config: TentCalcConfig, calc_fn: Callable) -> str:
    """Generate the filename from config and calculation function."""
    filename = (
        f"tents_ts_{config.LAG}Lag_fn={calc_fn.__name__}"
        f"_{config.WINDOW_SIZE}window_{config.WINDOW_STEP}step"
        f"sourcecol={config.on_column.value}"
    )
    return filename


MIN_TICKS_PER_DAY = 200  # Minimum data in a trading day for robust entropy calculation.

DATA_PATH = Path("analysis/data/full_futures/")

RETURNS_DATA = DATA_PATH / Path("futures_returns.csv")

TE_DATA_PATH = DATA_PATH / Path("transfer_entropies_bootstrap.csv")
PLOT_PATH = Path("analysis/plots/full_futures/")
PLOT_PATH.mkdir(exist_ok=True)

NETWORK_PATH = PLOT_PATH / Path("networks")
NETWORK_PATH.mkdir(exist_ok=True)
