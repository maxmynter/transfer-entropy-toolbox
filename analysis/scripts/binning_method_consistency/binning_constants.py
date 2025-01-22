"""Constants and configuration for binning analysis."""

from pathlib import Path

import numpy as np
from metric_enum import Metric

from te_toolbox.binning import (
    aic_bins,
    bic_bins,
    cv_bins,
    doanes_bins,
    freedman_diaconis_bins,
    knuth_bins,
    rice_rule_bins,
    scotts_rule_bins,
    shimazaki_bins,
    sqrt_n_bins,
    sturges_bins,
)
from te_toolbox.systems import TentMap

# Metrics for Analysis
METRICS = [Metric.TE, Metric.LOGNTE]  # [Metric.TE, Metric.HNTE, Metric.LOGNTE]

# Analysis Parameters
SEED = 42
EPS = 0.5
N_TRANSIENT = 10**5
N_MAPS = 50
LAG = 1
MIN_SAMPLE = 50
MAX_SAMPLE = 1.5 * 10**4
N_SAMPLE = 100
SAMPLE_SIZES = [int(i) for i in np.geomspace(MIN_SAMPLE, MAX_SAMPLE, N_SAMPLE)]

# Map Configuration
DEFAULT_MAP = TentMap(r=2)

# Binning Methods
BINNING_METHODS = {
    "AIC": aic_bins,
    "BIC": bic_bins,
    "Doane": doanes_bins,
    "Freedman-Diaconis": freedman_diaconis_bins,
    "Knuth": knuth_bins,
    "Rice": rice_rule_bins,
    "Scott": scotts_rule_bins,
    "Shimazaki": shimazaki_bins,
    "Sqrt-n": sqrt_n_bins,
    "Sturges": sturges_bins,
    "Cross-Validation": cv_bins,
    **{"Max " + str(metric): metric.maximising_bins for metric in METRICS},
}

# File naming patterns
RESULTS_FILE_PATTERN = "binning_results_size_{}.pkl"
CRITERION_BINS_PATTERN = "bin_criterion_size_comparison_{}.png"

# Directory Structure
BASE_DIR = Path("analysis")
DATA_DIR = BASE_DIR / "data" / "binning_method_consistency"
PLOT_DIR = BASE_DIR / "plots" / "binning_method_consistency"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Plot Configuration
PLOT_STYLES = {
    "figure.figsize": (15, 8),
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "stix",
    "text.usetex": True,
}
