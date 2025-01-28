"""Test the binning methods."""

from math import lgamma

import numpy as np
import numpy.typing as npt
import pytest

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
    aic_cost,
    bic_bins,
    bic_cost,
    knuth_bins,
    knuth_cost,
    shimazaki_bins,
    shimazaki_cost,
    small_sample_akaike_bins,
)
from tests.conftest import NUMERIC_TOLERANCE


# Common test data
@pytest.fixture
def uniform_data() -> npt.NDArray[np.float64]:
    """Generate uniform test data."""
    np.random.seed(42)
    return np.random.uniform(0, 10, 1000)


@pytest.fixture
def normal_data() -> npt.NDArray[np.float64]:
    """Generate normal test data."""
    np.random.seed(42)
    return np.random.normal(5, 2, 1000)


@pytest.fixture
def bimodal_data() -> npt.NDArray[np.float64]:
    """Generate bimodal test data."""
    np.random.seed(42)
    return np.concatenate(
        [np.random.normal(2, 0.5, 500), np.random.normal(8, 0.5, 500)]
    )


def test_bin_edge_properties(uniform_data: npt.NDArray[np.float64]):
    """Test basic properties that all binning methods should satisfy."""
    binning_methods = [
        sqrt_n_bins,
        freedman_diaconis_bins,
        sturges_bins,
        rice_rule_bins,
        doanes_bins,
        scotts_rule_bins,
        aic_bins,
        bic_bins,
        knuth_bins,
        shimazaki_bins,
    ]
    min_edges = 3

    for method in binning_methods:
        bins = method(uniform_data)

        # Test basic properties
        assert isinstance(bins, np.ndarray)
        assert bins.dtype == np.float64
        assert len(bins) >= min_edges
        assert np.all(np.diff(bins) > 0)
        assert bins[0] <= uniform_data.min()
        assert bins[-1] >= uniform_data.max()


def test_sqrt_n_bins(normal_data: npt.NDArray[np.float64]):
    """Test number of bins sqrt N."""
    bins = sqrt_n_bins(normal_data)
    expected = int(np.ceil(np.sqrt(len(normal_data))))
    assert len(bins) - 1 == expected


def test_sturges_formula(normal_data: npt.NDArray[np.float64]):
    """Test Sturges' formula specific properties."""
    bins = sturges_bins(normal_data)
    expected_bins = int(np.ceil(np.log2(len(normal_data))) + 1)
    assert len(bins) - 1 == expected_bins


def test_rice_rule(normal_data: npt.NDArray[np.float64]):
    """Test Rice rule specific properties."""
    bins = rice_rule_bins(normal_data)
    expected_bins = int(np.ceil(2 * np.cbrt(len(normal_data))))
    assert len(bins) - 1 == expected_bins


def test_doanes_formula_calculation():
    """Test Doane's formula calculation matches manual calculation."""
    # Create simple test data
    np.random.seed(42)
    data = np.random.normal(0, 1, 100).astype(np.float64)

    n = len(data)
    g1 = np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    sg1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))

    expected_nbins = int(1 + np.log2(n) + np.log2(1 + abs(g1) / sg1))

    actual_bins = doanes_bins(data)
    actual_nbins = len(actual_bins) - 1

    assert actual_nbins == expected_nbins


def test_doanes_formula_skewness(normal_data: npt.NDArray[np.float64]):
    """Test Doane's formula handles skewness correctly."""
    # Normal distribution should give similar results to Sturges
    bins_doane = doanes_bins(normal_data)
    bins_sturges = sturges_bins(normal_data)
    max_deviation = 2
    assert abs(len(bins_doane) - len(bins_sturges)) <= max_deviation

    # Skewed data should give more bins than Sturges
    skewed_data = np.exp(normal_data)  # Log-normal distribution is skewed
    bins_doane_skewed = doanes_bins(skewed_data)
    bins_sturges_skewed = sturges_bins(skewed_data)
    assert len(bins_doane_skewed) > len(bins_sturges_skewed)


def test_freedman_diaconis_calculation():
    """Test Freedman-Diaconis bin width calculation matches manual calculation."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    expected_width = 2 * iqr / np.cbrt(n)  # 2 * 3.5 / 2

    bins = freedman_diaconis_bins(data)
    actual_width = bins[1] - bins[0]

    np.testing.assert_allclose(actual_width, expected_width)

    assert bins[0] <= data.min()
    assert bins[-1] >= data.max()


def test_scotts_rule_normal_optimal(normal_data: npt.NDArray[np.float64]):
    """Test Scott's rule is optimal for normal distribution."""
    bins = scotts_rule_bins(normal_data)
    h = bins[1] - bins[0]

    expected_h = 3.49 * np.std(normal_data) / np.cbrt(len(normal_data))
    assert np.abs(h - expected_h) < NUMERIC_TOLERANCE


def test_statistical_criteria(uniform_data: npt.NDArray[np.float64]):
    """Test statistical criteria based methods."""
    methods = [aic_bins, bic_bins, knuth_bins, shimazaki_bins]
    min_bins = 2
    deviation_tolerance = 0.1

    for method in methods:
        bins = method(uniform_data)
        hist, _ = np.histogram(uniform_data, bins)

        # Basic sanity checks
        assert len(hist) >= min_bins
        assert np.all(hist > 0)  # No empty bins for uniform data

        # Check if histogram is roughly uniform
        expected_count = len(uniform_data) / len(hist)
        rel_deviation = np.abs(hist - expected_count) / expected_count
        assert np.mean(rel_deviation) < deviation_tolerance


def test_knuth_cost_calculation():
    """Test Knuth's rule cost function calculation matches formula."""
    # Simple test case where we can manually verify
    hist = np.array([10, 15, 5], dtype=np.int64)  # 3 bins with known counts
    bins = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    # Calculate components manually:
    n = np.sum(hist)  # total count = 30
    m = len(hist)  # number of bins = 3

    expected_cost = (
        n * np.log(m)  # 30 * log(3)
        + lgamma(m / 2)  # lgamma(1.5)
        - lgamma(n + m / 2)  # lgamma(31.5)
        - m * lgamma(0.5)  # 3 * lgamma(0.5)
        + sum(
            lgamma(count + 0.5) for count in hist
        )  # sum of lgamma for [10.5, 15.5, 5.5]
    )

    actual_cost = knuth_cost(hist, bins)
    np.testing.assert_allclose(actual_cost, expected_cost)


def test_shimazaki_cost_calculation():
    """Test Shimazaki-Shinomoto cost function calculation."""
    hist = np.array([10, 15, 5], dtype=np.int64)
    bins = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    # Manual calculation from thesis formula:
    n = np.sum(hist)
    h = bins[1] - bins[0]  # bin width
    mean = np.mean(hist)
    var = np.var(hist)

    expected_cost = (2 * mean - var) / (h * n) ** 2

    actual_cost = shimazaki_cost(hist, bins)
    np.testing.assert_allclose(actual_cost, expected_cost)


def test_aic_cost_calculation():
    """Test AIC cost function calculation."""
    hist = np.array([10, 15, 5], dtype=np.int64)
    bins = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    # Manual calculation from thesis formula:
    n = np.sum(hist)
    h = bins[1] - bins[0]
    m = len(hist)
    nonzero_hist = hist[hist > 0]

    expected_cost = (
        m  # number of bins
        + n * np.log(n)  # constant term
        + n * np.log(h)  # bin width term
        - np.sum(nonzero_hist * np.log(nonzero_hist))  # entropy term
    )

    actual_cost = aic_cost(hist, bins)
    np.testing.assert_allclose(actual_cost, expected_cost)


def test_small_sample_akaike_correction():
    """Test that correction term behaves as expected with sample size."""
    # Large sample - correction should be minimal
    large_data = np.random.normal(0, 1, 1000).astype(np.float64)
    # Small sample - correction should be significant
    small_data = large_data[:20]

    small_aic = len(aic_bins(small_data))
    small_aicc = len(small_sample_akaike_bins(small_data))

    # For small samples, should use fewer bins than AIC
    assert small_aicc < small_aic


def test_bic_cost_calculation():
    """Test BIC cost function calculation."""
    hist = np.array([10, 15, 5], dtype=np.int64)
    bins = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    # Manual calculation from thesis formula:
    n = np.sum(hist)
    h = bins[1] - bins[0]
    m = len(hist)
    nonzero_hist = hist[hist > 0]

    expected_cost = (
        np.log(n) / 2 * m  # complexity penalty
        + n * np.log(n)  # constant term
        + n * np.log(h)  # bin width term
        - np.sum(nonzero_hist * np.log(nonzero_hist))  # entropy term
    )

    actual_cost = bic_cost(hist, bins)
    np.testing.assert_allclose(actual_cost, expected_cost)


def test_consistency():
    """Test consistency of binning methods with same input."""
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = data1.copy()

    # Test all non-clustering based bins. The latter have random initialisations.
    methods = [
        sqrt_n_bins,
        sturges_bins,
        rice_rule_bins,
        doanes_bins,
        scotts_rule_bins,
        aic_bins,
        bic_bins,
        knuth_bins,
        shimazaki_bins,
    ]

    for method in methods:
        bins1 = method(data1)
        bins2 = method(data2)
        np.testing.assert_array_almost_equal(bins1, bins2)
