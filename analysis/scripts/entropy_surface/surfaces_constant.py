"""Constants for the Surface Plots."""

from pathlib import Path

import numpy as np

from te_toolbox.systems import TentMap

EPS_DATA_DIR = Path("analysis/data/epsilon_scan/")
EPS_DATA_DIR.mkdir(exist_ok=True, parents=True)
SURFACE_PLOT_DIR = Path("analysis/plots/entropy_surface/")
SURFACE_PLOT_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42

N_TRANSIENT = 10**4
N_MAPS = 25
N_ITER = 10**4
LAG = 1

MIN_BINS = 2
MAX_BINS = 30
N_BINS = 3

MIN_POW = 2
LOGSPACE_BASE = 10
N_LENS = 3

bin_range = np.arange(MIN_BINS, MAX_BINS, int((MAX_BINS - MIN_BINS) / N_BINS))
length_range = np.logspace(
    MIN_POW, np.log10(N_ITER), base=LOGSPACE_BASE, num=N_LENS, dtype=int
)

maps = {
    # "LogisticMap(r=4)": LogisticMap(r=4),
    # "BellowsMap(r=5,b=6)": BellowsMap(r=5, b=6),
    # "ExponentialMap(r=4)": ExponentialMap(r=4),
    "TentMap(r=2)": TentMap(r=2),
}
