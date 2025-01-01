"""Test for conditional entropy measure."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from te_toolbox.entropies import joint_entropy
from tests.conftest import bin_generator


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


def test_jent_identical_variables():
    """Test joint entropy of identical variables: H(X,X) = H(X)."""
    data = np.array([[1, 1], [2, 2], [3, 3]])
    result = joint_entropy(data, bins=3)
    # Diagonal elements should be equal to single variable entropy
    assert_almost_equal(result[0, 0], result[1, 1])


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
