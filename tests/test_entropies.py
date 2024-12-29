"""Tests for entropy measures."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from te_toolbox.entropies import entropy


def test_entropy_uniform_distribution():
    """Test entropy calculation for uniform distribution."""
    data = np.array([1, 2, 3, 4, 5])
    result = entropy(data, bins=5)
    expected = -np.sum((np.ones(5) / 5) * np.log(np.ones(5) / 5))
    assert_almost_equal(result, expected)


def test_entropy_deterministic_distribution():
    """Test entropy calculation for deterministic distribution."""
    # All same values should have zero entropy
    data = np.array([1, 1, 1, 1, 1])
    result = entropy(data, bins=5)
    assert_almost_equal(result, 0.0)


def test_entropy_2d_data():
    """Test entropy calculation for 2D data."""
    data = np.array([[1, 4], [1, 5], [2, 6], [2, 7], [3, 8]])
    result = entropy(data, bins=3)
    assert len(result) == np.ndim(data), "Should return entropy for each variable"
    assert all(result >= 0), "Entropy should be non-negative"


def test_entropy_different_bin_specifications():
    """Test entropy calculation with different bin specifications."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    result1 = entropy(data, bins=3)
    result2 = entropy(data, bins=[3, 3])
    assert_array_almost_equal(result1, result2)


def test_entropy_error_handling():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        data_3d = np.zeros((2, 2, 2))
        entropy(data_3d, bins=2)


def test_entropy_zero_probability_handling():
    """Test handling of zero probabilities in histogram."""
    data = np.array([1, 1, 5, 5])
    result = entropy(data, bins=5)
    assert np.isfinite(result)  # Result should be finite
    assert result >= 0  # Entropy should be non-negative


def test_entropy_array_constitution_edge_cases():
    """Test edge cases."""
    data = np.array([1])
    result = entropy(data, bins=2)
    assert_almost_equal(result, 0.0)

    # Empty array should raise an error
    with pytest.raises(ValueError):
        entropy(np.array([]), bins=2)


def test_entropy_numerical_stability():
    """Test numerical stability with very large and small numbers."""
    data = np.array([1e-10, 1e10])
    result = entropy(data, bins=2)
    assert np.isfinite(result)  # Result should be finite
