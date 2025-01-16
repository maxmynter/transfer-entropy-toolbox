"""Constants for the epsilon scan."""

from pathlib import Path

import numpy as np

from te_toolbox.systems.maps import BellowsMap, ExponentialMap, LogisticMap, TentMap

EPS_DATA_DIR = Path("analysis/data/epsilon_scan/")
EPS_DATA_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42
N_TRANSIENT = 10**4
N_MAPS = 100
N_ITER = 3 * 10**3
LAG = 1
# Exclude eps=1 gives perfect self-explainability for Tent map.
EPSILONS = np.linspace(0, 1, 50)[:-1]
N_BINS = np.arange(2, 31)
RELATIVE_NOISE_AMPLITUDE = 0.0
PLOT_PATH = Path("analysis/plots/epsilon_scan/")
PLOT_PATH.mkdir(exist_ok=True, parents=True)


maps = {
    "LogisticMap(r=4)": LogisticMap(r=4),
    "BellowsMap(r=5,b=6)": BellowsMap(r=5, b=6),
    "ExponentialMap(r=4)": ExponentialMap(r=4),
    "TentMap(r=2)": TentMap(r=2),
}
