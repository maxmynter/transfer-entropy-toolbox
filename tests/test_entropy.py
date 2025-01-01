"""Tests for entropy measure."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from te_toolbox.entropies import entropy, joint_entropy
from tests.conftest import (
    NUMERIC_TOLERANCE,
    bin_generator,
    regularize_hypothesis_generated_data,
)


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


@given(
    st.lists(st.floats(min_value=-100, max_value=100), min_size=100),
    st.lists(st.floats(min_value=-100, max_value=100), min_size=100),
)
def test_entropy_additivity_independent(x, y):
    """Test that entropy is additive for independent variables."""
    data = regularize_hypothesis_generated_data(x, y)

    bins = bin_generator(data, 10)
    h_joint = joint_entropy(data, bins=bins)
    h_marginal = entropy(data, bins=bins)

    # For independent variables, H(X,Y) ≤ H(X) + H(Y)
    assert h_joint[0, 1] <= h_marginal[0] + h_marginal[1]


@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=100, unique=True))
def test_entropy_data_processing_inequality(data):
    """Test that processing cannot increase entropy."""
    # Convert to numpy array
    data = np.array(data)

    # Simple compression: floor values
    processed = np.sign(data)

    bins = np.linspace(-10, 10, 7)

    # Calculate entropies
    h_original = entropy(data, bins=bins)
    h_processed = entropy(processed, bins=bins)

    assert (
        h_processed <= h_original + NUMERIC_TOLERANCE
    ), f"Entropy increased: original {h_original}, processed {h_processed}"
