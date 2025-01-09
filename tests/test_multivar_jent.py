"""Tests for the multivariate joint entropy."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_almost_equal

from te_toolbox.entropies import entropy, multivar_joint_entropy
from tests.conftest import NUMERIC_TOLERANCE


def test_multivar_jent_known_distribution():
    """Test multivariate joint entropy with known probability distribution.

    Create a distribution where X,Y,Z are all binary and independent:
    P(X=0,Y=0,Z=0) = 1/8, P(X=0,Y=0,Z=1) = 1/8, etc.
    This gives maximum joint entropy for 3 binary variables:
    H(X,Y,Z) = -8 * (1/8 * log(1/8)) = 3*log(2)
    """
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    data = np.column_stack([x, y, z])

    result = multivar_joint_entropy(data, bins=2)
    expected = 3 * np.log(2)  # Maximum entropy for 3 binary variables
    assert_almost_equal(result, expected)


def test_multivar_jent_chain_rule():
    """Test that H(X,Y,Z) ≥ H(X,Y) ≥ H(X)."""
    x = np.array([1, 1, 2, 2])
    y = np.array([1, 2, 1, 2])
    z = np.array([1, 2, 2, 1])
    data_3d = np.column_stack([x, y, z])
    data_2d = np.column_stack([x, y])

    h_xyz = multivar_joint_entropy(data_3d, bins=2)
    h_xy = multivar_joint_entropy(data_2d, bins=2)
    h_x = entropy(x.reshape(-1), bins=2)

    assert h_xyz >= h_xy - NUMERIC_TOLERANCE
    assert h_xy >= h_x - NUMERIC_TOLERANCE


def test_multivar_jent_symmetry():
    """Test that variable order doesn't matter."""
    x = np.array([1, 1, 2, 2])
    y = np.array([1, 2, 1, 2])
    z = np.array([2, 1, 2, 1])

    data_xyz = np.column_stack([x, y, z])
    data_zyx = np.column_stack([z, y, x])

    result_xyz = multivar_joint_entropy(data_xyz, bins=2)
    result_zyx = multivar_joint_entropy(data_zyx, bins=2)

    assert_almost_equal(result_xyz, result_zyx)


def test_multivar_jent_different_bin_specs():
    """Test different bin specifications."""
    data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    result1 = multivar_joint_entropy(data, bins=3)
    result2 = multivar_joint_entropy(data, bins=[3, 3, 3])
    assert_almost_equal(result1, result2)

    # Test with different numbers of bins per variable
    result3 = multivar_joint_entropy(data, bins=[2, 3, 4])
    assert result3 > 0  # Just verify it runs and gives reasonable output


@given(
    st.lists(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=3, max_size=3),
        min_size=10,
        max_size=50,
    )
)
def test_multivar_jent_hypothesis(data):
    """Property-based testing for multivariate joint entropy."""
    data = np.array(data)
    bins = 5
    result = multivar_joint_entropy(data, bins=bins)
    assert result >= 0
    # Joint entropy should be greater than individual entropies
    for i in range(data.shape[1]):
        h_i = entropy(data[:, i], bins=bins)
        assert result >= h_i - NUMERIC_TOLERANCE
