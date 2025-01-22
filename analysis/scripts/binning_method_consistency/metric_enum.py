"""Contains the (N)TE metric enum for nice parametrization."""

from enum import Enum, auto

from max_entropy_bins import max_logntent_bins, max_ntent_bins, max_tent_bins

from te_toolbox.entropies.transfer import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)


class Metric(Enum):
    """Enum holding (N)TE name parametrization."""

    TE = auto()
    HNTE = auto()
    LOGNTE = auto()

    def __str__(self):
        """Represent nicely formatted string of (N)TE values."""
        return {Metric.TE: "TE", Metric.HNTE: "NTE", Metric.LOGNTE: "logNTE"}[self]

    @property
    def compute(self):
        """Calculate the corresponding measure."""
        return {
            Metric.TE: transfer_entropy,
            Metric.HNTE: normalized_transfer_entropy,
            Metric.LOGNTE: logn_normalized_transfer_entropy,
        }[self]

    @property
    def maximising_bins(self):
        """Get the maximising bins per measure."""
        return {
            Metric.TE: max_tent_bins,
            Metric.HNTE: max_ntent_bins,
            Metric.LOGNTE: max_logntent_bins,
        }[self]
