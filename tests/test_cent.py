"""Test for joint entropy measure."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from te_toolbox.entropies import conditional_entropy, joint_entropy


def test_cent_nonnegativity():
    """Test that conditional entropy is non-negative."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    result = conditional_entropy(data, bins=3)
    assert np.all(result >= 0)


def test_cent_relation_to_joint():
    """Test relation H(X|Y) = H(X,Y) - H(Y)."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    h_xy = joint_entropy(data, bins=3)
    h_x = np.diag(h_xy)  # Marginal entropies are on diagonal
    result = conditional_entropy(data, bins=3)

    # Test H(X|Y) = H(X,Y) - H(Y) for each pair
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            expected = h_xy[i, j] - h_x[j]
            assert_almost_equal(result[i, j], expected)


def test_cent_self_conditioning():
    """Test that conditioning on itself gives zero: H(X|X) = 0."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    result = conditional_entropy(data, bins=3)
    assert_array_almost_equal(np.diag(result), np.zeros(data.shape[1]))


def test_cent_chain_rule():
    """Test chain rule H(X,Y) = H(X) + H(Y|X)."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    h_xy = joint_entropy(data, bins=3)
    h_x = np.diag(h_xy)  # Marginal entropies
    result = conditional_entropy(data, bins=3)

    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            assert_almost_equal(h_xy[i, j], h_x[i] + result[j, i])


def test_cent_error_handling():
    """Test error handling in conditional entropy."""
    # Test 1D input
    with pytest.raises(ValueError):
        conditional_entropy(np.array([1, 2, 3]), bins=3)

    # Test empty input
    with pytest.raises(ValueError):
        conditional_entropy(np.array([[]]), bins=3)
