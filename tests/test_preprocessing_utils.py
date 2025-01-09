"""Tests for preprocessing utilities."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_almost_equal, assert_array_equal

from te_toolbox.preprocessing import noisify, remap_to


def test_noisify_shape_preservation():
    """Test if noisify preserves input shape."""
    data = np.random.normal(0, 1, (100, 2))
    noisy = noisify(data, "normal")
    assert noisy.shape == data.shape


def test_noisify_different_distributions():
    """Test noisify with different built-in distributions."""
    data = np.random.normal(0, 1, (100, 2))
    distributions = ["normal", "uniform", "poisson"]

    for dist in distributions:
        noisy = noisify(data, dist)
        assert noisy.shape == data.shape
        # Check that noise was actually added
        assert not np.array_equal(noisy, data)


def test_noisify_custom_sampler():
    """Test noisify with custom noise sampler."""
    data = np.random.normal(0, 1, (100, 2))
    noisy = noisify(data, lambda size, **kwargs: np.random.exponential(size=size))
    assert noisy.shape == data.shape
    assert not np.array_equal(noisy, data)


def test_noisify_invalid_distribution():
    """Test noisify with invalid distribution specification."""
    data = np.random.normal(0, 1, (100, 2))

    with pytest.raises(ValueError):
        noisify(data, "invalid_distribution")


@given(
    st.integers(min_value=10, max_value=1000),
    st.integers(min_value=1, max_value=10),
    st.floats(min_value=0.1, max_value=2.0),
)
def test_noisify_properties(n_samples, n_vars, amplitude):
    """Test noisify properties with different parameters."""
    data = np.random.normal(0, 1, (n_samples, n_vars))
    original_mean = np.mean(data)

    noisy = noisify(data, "normal", amplitude=amplitude)

    assert noisy.shape == (n_samples, n_vars)
    # Mean should change due to noise, but not drastically
    assert np.abs(np.mean(noisy) - original_mean) < amplitude * 2


def test_remap_to_shape_mismatch():
    """Test remap_to with mismatched shapes."""
    data = np.random.normal(0, 1, (100, 2))
    distribution = np.random.normal(0, 1, (100, 3))

    with pytest.raises(ValueError):
        remap_to(data, distribution)


def test_remap_to_preserves_ranks():
    """Test if remap_to preserves rank ordering of original data."""
    data = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    distribution = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    result = remap_to(data, distribution)

    # Check rank preservation for each column
    for i in range(data.shape[1]):
        original_ranks = np.argsort(np.argsort(data[:, i]))
        remapped_ranks = np.argsort(np.argsort(result[:, i]))
        assert_array_equal(original_ranks, remapped_ranks)


def test_remap_to_uses_distribution_values():
    """Test if remap_to uses values from the target distribution."""
    data = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    distribution = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    result = remap_to(data, distribution)

    # Check if all values in result come from distribution
    for i in range(data.shape[1]):
        assert set(result[:, i]) == set(distribution[:, i])


@given(
    st.integers(min_value=10, max_value=1000),
    st.integers(min_value=1, max_value=10),
)
def test_remap_to_properties(n_samples, n_vars):
    """Test remap_to properties with different dimensions."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (n_samples, n_vars))
    distribution = rng.normal(5, 2, (n_samples, n_vars))

    result = remap_to(data, distribution, rng)

    assert result.shape == (n_samples, n_vars)
    # Check that output uses distribution values
    for i in range(n_vars):
        assert_array_almost_equal(np.sort(result[:, i]), np.sort(distribution[:, i]))


def test_remap_to_deterministic():
    """Test if remap_to is deterministic without RNG."""
    data = np.random.normal(0, 1, (100, 2))
    distribution = np.random.normal(0, 1, (100, 2))

    result1 = remap_to(data, distribution)
    result2 = remap_to(data, distribution)

    assert_array_equal(result1, result2)


def test_remap_to_with_ties():
    """Test remap_to behavior with tied values."""
    data = np.array([[1.0, 2.0], [1.0, 1.0], [2.0, 3.0]])
    distribution = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

    # Without RNG, ties should be handled consistently
    result1 = remap_to(data, distribution)
    result2 = remap_to(data, distribution)
    assert_array_equal(result1, result2)

    # With RNG, tied values might be handled differently
    rng = np.random.default_rng(42)
    result_rng = remap_to(data, distribution, rng)
    # Should still maintain overall structure
    assert set(result_rng[:, 0]) == set(distribution[:, 0])


@given(
    st.integers(min_value=100, max_value=1000),
    st.integers(min_value=2, max_value=5),
)
def test_remap_to_distribution_preservation(n_samples, n_vars):
    """Test if remap_to preserves the exact values from distribution."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (n_samples, n_vars))
    distribution = rng.normal(5, 2, (n_samples, n_vars))

    result = remap_to(data, distribution, rng)

    # For each column, check if the values match exactly with distribution
    for i in range(n_vars):
        assert set(np.round(result[:, i], 10)) == set(np.round(distribution[:, i], 10))
