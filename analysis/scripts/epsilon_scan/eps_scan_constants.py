"""Constants for the epsilon scan."""

from pathlib import Path

import numpy as np

EPS_DATA_DIR = Path("analysis/data/epsilon_scan/")
EPS_DATA_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42
N_TRANSIENT = 10**4
N_MAPS = 100
N_ITER = 3 * 10**3
LAG = 1
EPSILONS = np.linspace(0, 1, 20)
N_BINS = np.arange(2, 31)
RELATIVE_NOISE_AMPLITUDE = 0.0
PLOT_PATH = Path("analysis/plots/epsilon_scan/")
PLOT_PATH.mkdir(exist_ok=True, parents=True)
