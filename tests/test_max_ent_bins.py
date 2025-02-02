"""Test for the maximum entropy binning criterion."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from te_toolbox.binning.entropy_maximising import (
    max_logntent_bins,
    max_ntent_bins,
    max_tent,
    max_tent_bins,
)
from te_toolbox.entropies import transfer_entropy

MIN_EDGES = 3


@pytest.fixture
def simple_coupled_data():
    """Create simple coupled data where we know transfer entropy exists."""
    n_samples = 1000
    x = np.random.uniform(0, 1, n_samples)
    y = 0.7 * np.roll(x, 1) + 0.3 * np.random.uniform(0, 1, n_samples)
    y[0] = np.random.uniform(0, 1)
    return np.column_stack((x, y))


@pytest.fixture
def independent_data():
    """Create independent data where transfer entropy should be minimal."""
    return np.random.uniform(0, 1, (1000, 2))


def test_max_tent_returns_valid_bins():
    """Test that max_tent returns valid bins."""
    data = np.random.uniform(0, 1, (100, 2))
    bins = max_tent(transfer_entropy, data)
    assert isinstance(bins, np.ndarray)
    assert len(bins) > MIN_EDGES
    assert len(bins) < len(data)
    assert bins.dtype == np.float64


def test_max_tent_coupled_vs_independent(simple_coupled_data, independent_data):
    """Test bin count difference between coupled/independent data."""
    coupled_bins = max_tent(transfer_entropy, simple_coupled_data)
    indep_bins = max_tent(transfer_entropy, independent_data)
    assert len(coupled_bins) >= len(indep_bins)


def test_max_tent_bins_matches_direct_call():
    """Test wrapper matches direct call."""
    data = np.random.uniform(0, 1, (100, 2))
    direct_bins = max_tent(transfer_entropy, data)
    wrapper_bins = max_tent_bins(data)
    assert_array_equal(direct_bins, wrapper_bins)


def test_different_entropy_measures_give_different_bins(simple_coupled_data):
    """Test different entropy measures produce different bins."""
    te_bins = max_tent_bins(simple_coupled_data)
    nte_bins = max_ntent_bins(simple_coupled_data)
    lognte_bins = max_logntent_bins(simple_coupled_data)
    bins_list = [len(b) for b in [te_bins, nte_bins, lognte_bins]]
    assert len(set(bins_list)) > 1


def test_max_tent_handles_edge_cases():
    """Test edge case handling."""
    # Minimal data
    min_data = np.random.uniform(0, 1, (4, 2))
    bins = max_tent(transfer_entropy, min_data)
    assert isinstance(bins, np.ndarray)
    assert len(bins) >= MIN_EDGES


@pytest.mark.parametrize("data_size", [(100, 2), (1000, 2), (5000, 2)])
def test_max_tent_scales_with_data_size(data_size):
    """Test bin count scaling."""
    data = np.random.uniform(0, 1, data_size)
    bins = max_tent(transfer_entropy, data)
    assert MIN_EDGES < len(bins) < len(data)
    assert len(bins) < 2 * np.sqrt(len(data))
