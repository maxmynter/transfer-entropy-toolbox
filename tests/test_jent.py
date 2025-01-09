"""Test for conditional entropy measure."""

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


def test_jent_known_distribution():
    """Test joint entropy with a known joint probability distribution.

    Create a joint distribution where:
    P(X=0,Y=0) = 0.25  P(X=0,Y=1) = 0.25
    P(X=1,Y=0) = 0.25  P(X=1,Y=1) = 0.25

    This gives us maximum joint entropy for 2 binary variables:
    H(X,Y) = -sum(p(x,y) * log(p(x,y))) = -4 * (0.25 * log(0.25)) = 2*log(2)
    """
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    data = np.column_stack([x, y])

    result = joint_entropy(data, bins=2)

    # Expected: maximum joint entropy for 2 binary variables
    expected = 2 * np.log(2)  # = -4 * (0.25 * log(0.25))

    assert_almost_equal(result[0, 1], expected)


@given(
    st.lists(
        st.lists(st.floats(min_value=-1000, max_value=1000), min_size=2, max_size=2),
        min_size=2,
    )
)
def test_jent_symmetry_property(data):
    """Test joint entropy symmetry property with random data."""
    data = np.array(data)
    bins = bin_generator(data, 10)
    result = joint_entropy(data, bins=bins)
    assert_array_almost_equal(result, result.T)


def test_jent_independent_variables():
    """Test joint entropy of independent variables: H(X,Y) = H(X) + H(Y)."""
    # Create independent variables
    x = np.array([1, 1, 2, 2])  # p(X=1) = p(X=2) = 0.5
    y = np.array([1, 2, 1, 2])  # p(Y=1) = p(Y=2) = 0.5
    data = np.column_stack([x, y])

    result = joint_entropy(data, bins=2)
    # H(X) + H(Y) for uniform distributions
    expected = 2 * (-0.5 * np.log(0.5) - 0.5 * np.log(0.5))
    assert_almost_equal(result[0, 1], expected)


@given(
    st.lists(
        st.floats(min_value=-1, max_value=1), unique=True, min_size=15, max_size=50
    )
)
def test_jent_identical_variables(x):
    """Test joint entropy of identical variables: H(X,X) = H(X)."""
    data = np.array([x])
    bins = np.linspace(np.min(x), np.max(x), 3)
    h_x = entropy(data, bins=bins)
    h_xx = np.diag(joint_entropy(data, bins=bins))
    # Diagonal elements should be equal to single variable entropies
    assert np.all(
        np.abs(h_x - h_xx) <= NUMERIC_TOLERANCE
    ), f"H(x) = H(x,x) but found {h_x} != {h_xx}"


@given(
    st.lists(
        st.floats(min_value=-1, max_value=1), min_size=15, max_size=50, unique=True
    ),
    st.lists(
        st.floats(min_value=-1, max_value=1), min_size=15, max_size=50, unique=True
    ),
)
def test_jent_max_property(x, y):
    """Test joint greater individual entropies, H(X,Y) >= max(H(X), H(Y))."""
    data = regularize_hypothesis_generated_data(x, y)
    bins = np.linspace(-1, 1, 3)

    h_x = entropy(data[0], bins=bins)
    h_y = entropy(data[1], bins=bins)
    h_xy = joint_entropy(data, bins=bins)[0, 1]

    assert h_xy >= max(h_x, h_y) - NUMERIC_TOLERANCE


def test_jent_different_bin_specs():
    """Test joint entropy with different bin specifications."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    result1 = joint_entropy(data, bins=3)
    result2 = joint_entropy(data, bins=[3, 3])
    assert_array_almost_equal(result1, result2)


def test_jent_error_handling():
    """Test error handling in joint entropy."""
    # Test 1D input
    with pytest.raises(ValueError):
        joint_entropy(np.array([1, 2, 3]), bins=3)

    # Test empty input
    with pytest.raises(ValueError):
        joint_entropy(np.array([[]]), bins=3)
