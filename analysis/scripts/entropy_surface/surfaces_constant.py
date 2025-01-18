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
N_MAPS = 15
N_ITER = 10000
LAG = 1

MIN_BINS = 2
MAX_BINS = 80
BIN_STEP = 2

N_BINS = int((MAX_BINS - MIN_BINS) / BIN_STEP)

MIN_LEN = 30
LEN_STEP = 10
N_LENS = int((N_ITER - MIN_LEN) / LEN_STEP)

bin_range = np.arange(MIN_BINS, MAX_BINS, BIN_STEP)
length_range = np.arange(MIN_LEN, N_ITER, LEN_STEP)

maps = {
    "TentMap(r=2)": TentMap(r=2),
}
