"""Tests for mutual information measure."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_almost_equal

from te_toolbox.entropies import entropy, mutual_information


def test_mi_symmetric():
    """Test MI symmetry: I(X,Y) = I(Y,X)."""
    data_l = np.array([[1, 4], [2, 5], [3, 6]])
    data_r = np.array([[4, 1], [5, 2], [6, 3]])
    result_l = mutual_information(data_l, bins=3)
    result_r = mutual_information(data_r, bins=3)
    assert_array_almost_equal(result_l, result_r)


@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=25, unique=True))
def test_mi_self_information(x):
    """Test MI(X,X) equals H(X)."""
    data = np.column_stack([x, x])
    mi = mutual_information(data, bins=5, norm=False)
    h = entropy(data, bins=5)
    assert np.abs(mi[0, 1] - h[0]) < 1e-6  # noqa: PLR2004  # Floating point comparison with absolute tolerance


def test_mi_error_handling():
    """Test error handling."""
    with pytest.raises(ValueError):
        mutual_information(np.array([1, 2, 3]), bins=3)


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_infinity=False, allow_nan=False),
        min_size=25,
        unique=True,
    )
)
def test_mi_bounds(x):
    """Test normalized MI is between 0 and 1."""
    data = np.column_stack([x, x])
    mi = mutual_information(data, bins=3, norm=True)
    assert np.all(~np.isnan(mi) & (mi >= 0) & (mi <= 1))


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_infinity=False, allow_nan=False),
        min_size=25,
    ).filter(lambda x: max(x) - min(x) > 0 and len(set(x)) > 2)  # noqa: PLR2004 # Need non-constant sequence for binning
)
def test_mi_independent_variables(x):
    """Test self MI equals entropy."""
    data = np.column_stack([x, x])
    mi = mutual_information(data, bins=10, norm=False)
    h = entropy(data, bins=10)
    assert_array_almost_equal(mi[0, 1], h[0])


@given(st.integers(min_value=1))
def test_mi_zero_for_independent(seed):
    """Test MI=0 for independent variables."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100, 100, 1000)
    y = rng.uniform(-100, 100, 1000)
    data = np.column_stack([x, y])
    mi = mutual_information(data, bins=10, norm=True)
    assert mi[0, 1] < 0.1  # noqa: PLR2004 # Allow for 10% of noise induced MI.


@given(st.integers(min_value=2, max_value=20))
def test_mi_different_bin_numbers(n_bins):
    """Test MI stability with different bin numbers."""
    x = np.linspace(0, 10, 100)
    data = np.column_stack([x, x])
    mi = mutual_information(data, bins=n_bins, norm=True)
    assert 0.99 < mi[0, 1] <= 1.0 + 10e-10  # noqa: PLR2004  # Should be close to 1 for identical series


def test_mi_noise_decreases_mi():
    """Test that adding noise decreases MI."""
    x = np.linspace(0, 10, 100)
    y1 = x + np.random.normal(0, 0.1, 100)
    y2 = x + np.random.normal(0, 1.0, 100)
    mi1 = mutual_information(np.column_stack([x, y1]), bins=10, norm=True)
    mi2 = mutual_information(np.column_stack([x, y2]), bins=10, norm=True)
    assert mi1[0, 1] > mi2[0, 1]
