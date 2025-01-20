"""Discretization and binning methods and utilities."""

from importlib.util import find_spec

from te_toolbox.binning.rules import (
    doanes_bins,
    freedman_diaconis_bins,
    rice_rule_bins,
    scotts_bins,
    sqrt_n_bins,
    sturges_bins,
)
from te_toolbox.binning.statistical import (
    aic_bins,
    bic_bins,
    knuth_bins,
    shimazaki_bins,
    small_sample_akaike_bins,
)

HAS_SKLEARN = find_spec("sklearn") is not None


if HAS_SKLEARN:
    from te_toolbox.binning.clustering import (
        agglomerative_bins,
        dbscan_bins,
        kmeans_bins,
        meanshift_bins,
    )

__all__ = [
    "HAS_SKLEARN",
    "agglomerative_bins",
    "aic_bins",
    "bic_bins",
    "dbscan_bins",
    "doanes_bins",
    "freedman_diaconis_bins",
    "kmeans_bins",
    "knuth_bins",
    "meanshift_bins",
    "rice_rule_bins",
    "scotts_bins",
    "shimazaki_bins",
    "small_sample_akaike_bins",
    "sqrt_n_bins",
    "sturges_bins",
]
