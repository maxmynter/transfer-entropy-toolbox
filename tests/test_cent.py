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


def test_conditional_entropy_known_distribution():
    """Test conditional entropy with independent variables."""
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    data = np.column_stack([x, y])
    result = conditional_entropy(data, bins=2)
    assert_almost_equal(result[1, 0], np.log(2))  # H(Y|X)
    assert_almost_equal(result[0, 1], np.log(2))  # H(X|Y)
    assert_almost_equal(result[0, 0], 0.0)  # H(X|X)
    assert_almost_equal(result[1, 1], 0.0)  # H(Y|Y)


def test_cent_with_dependent_distribution():
    """Test conditional entropy with completely dependent variables."""
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 1, 0, 1])
    data = np.column_stack([x, y])
    result = conditional_entropy(data, bins=2)
    assert_almost_equal(result[1, 0], 0.0)  # H(Y|X)
    assert_almost_equal(result[0, 1], 0.0)  # H(X|Y)
    assert_almost_equal(result[0, 0], 0.0)  # H(X|X)
    assert_almost_equal(result[1, 1], 0.0)  # H(Y|Y)


def test_cent_with_partial_dependency():
    """Test conditional entropy with partially dependent variable.

    X = [0,0,1,1], Y = [0,0,0,1]
    P(Y=0|X=0) = 1    P(Y=1|X=0) = 0
    P(Y=0|X=1) = 0.5  P(Y=1|X=1) = 0.5
    H(Y|X) = 1/2 * log(2)

    P(X=0|Y=0) = 2/3  P(X=1|Y=0) = 1/3
    P(X=0|Y=1) = 0    P(X=1|Y=1) = 1
    H(X|Y) = 3/4 * (-2/3 * log(2/3) - 1/3 * log(1/3))
    """
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 0, 1])
    data = np.column_stack([x, y])
    result = conditional_entropy(data, bins=2)
    y_given_x = 1 / 2 * np.log(2)
    x_given_y = 3 / 4 * (-2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3))
    assert_almost_equal(result[1, 0], y_given_x)  # H(Y|X)
    assert_almost_equal(result[0, 1], x_given_y)  # H(X|Y)
    assert_almost_equal(result[0, 0], 0.0)  # H(X|X)
    assert_almost_equal(result[1, 1], 0.0)  # H(Y|Y)
