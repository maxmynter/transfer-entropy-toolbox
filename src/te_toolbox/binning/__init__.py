"""Discretization and binning methods and utilities."""

from te_toolbox.binning.rules import (
    doanes_bins,
    freedman_diaconis_bins,
    rice_rule_bins,
    scotts_rule_bins,
    sqrt_n_bins,
    sturges_bins,
)
from te_toolbox.binning.statistical import (
    aic_bins,
    bic_bins,
    cv_bins,
    knuth_bins,
    shimazaki_bins,
    small_sample_akaike_bins,
)

__all__ = [
    "aic_bins",
    "bic_bins",
    "cv_bins",
    "doanes_bins",
    "freedman_diaconis_bins",
    "knuth_bins",
    "rice_rule_bins",
    "scotts_rule_bins",
    "shimazaki_bins",
    "small_sample_akaike_bins",
    "sqrt_n_bins",
    "sturges_bins",
]
