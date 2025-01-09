"""Constants for the epsilon scan."""

from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("data/")
SEED = 42
N_TRANSIENT = 10**4
N_MAPS = 100
N_ITER = 300
LAG = 1
EPSILONS = np.linspace(0, 1, 20)
N_BINS = np.arange(2, 31)
RELATIVE_NOISE_AMPLITUDE = 0.5
