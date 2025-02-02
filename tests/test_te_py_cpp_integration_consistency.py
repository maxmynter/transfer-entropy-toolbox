"""Test suite for backend consistency in entropy calculations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from te_toolbox.entropies import (
    conditional_entropy,
    entropy,
    joint_entropy,
    logn_normalized_transfer_entropy,
    transfer_entropy,
)
from te_toolbox.entropies.core import Backend, set_backend

# Constants
N_SAMPLES = 1000
N_VARS = 3
RTOL = 1e-5
ATOL = 1e-5


@pytest.fixture
def sample_3d_data():
    """Generate 3D sample data for testing."""
    np.random.seed(42)
    X = np.random.random(N_SAMPLES)  # noqa: N806 # Uppercase is fine for a dataset
    Y = X + np.random.random(N_SAMPLES)  # noqa: N806 # Uppercase is fine for a dataset
    Z = Y + np.random.random(N_SAMPLES)  # noqa: N806 # Uppercase is fine for a dataset
    return np.column_stack([X, Y, Z])


@pytest.fixture
def bins_3d():
    """Generate bin edges for 3D data."""
    return np.array([-0.5, 0.5, 1.5, 2.5, 3.5])


def test_transfer_entropy_3d_consistency(sample_3d_data, bins_3d):
    """Test consistency between Python and CPP backends for 3D transfer entropy."""
    data = sample_3d_data
    bins = bins_3d
    lag = 1

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_result_matrix = transfer_entropy(data, bins, lag)
    py_results = {}
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i != j:
                py_results[(i, j)] = transfer_entropy(data, bins, lag, at=(i, j))

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_result_matrix = transfer_entropy(data, bins, lag)
    cpp_results = {}
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i != j:
                cpp_results[(i, j)] = transfer_entropy(data, bins, lag, at=(i, j))

    # Compare full matrices
    assert_allclose(py_result_matrix, cpp_result_matrix, rtol=RTOL, atol=ATOL)

    # Compare individual pairs
    for key, _ in py_results.items():
        assert_allclose(py_results[key], cpp_results[key], rtol=RTOL, atol=ATOL)
        # Verify matrix entries match individual calculations
        i, j = key
        assert_allclose(py_result_matrix[i, j], py_results[key], rtol=RTOL, atol=ATOL)
        assert_allclose(cpp_result_matrix[i, j], cpp_results[key], rtol=RTOL, atol=ATOL)


def test_logn_normalized_te_3d_consistency(sample_3d_data, bins_3d):
    """Test consistency of Python/CPP backends for 3D normalized transfer entropy."""
    data = sample_3d_data
    bins = bins_3d
    lag = 1

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_result_matrix = logn_normalized_transfer_entropy(data, bins, lag)
    py_results = {}
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i != j:
                py_results[(i, j)] = logn_normalized_transfer_entropy(
                    data, bins, lag, at=(i, j)
                )

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_result_matrix = logn_normalized_transfer_entropy(data, bins, lag)
    cpp_results = {}
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i != j:
                cpp_results[(i, j)] = logn_normalized_transfer_entropy(
                    data, bins, lag, at=(i, j)
                )

    # Compare full matrices
    assert_allclose(py_result_matrix, cpp_result_matrix, rtol=RTOL, atol=ATOL)

    # Compare individual pairs
    for key, _ in py_results.items():
        assert_allclose(py_results[key], cpp_results[key], rtol=RTOL, atol=ATOL)
        # Verify matrix entries match individual calculations
        i, j = key
        assert_allclose(py_result_matrix[i, j], py_results[key], rtol=RTOL, atol=ATOL)
        assert_allclose(cpp_result_matrix[i, j], cpp_results[key], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_3d_te_different_lags(sample_3d_data, bins_3d, lag):
    """Test consistency between backends for different time lags."""
    data = sample_3d_data
    bins = bins_3d

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_result = transfer_entropy(data, bins, lag)

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_result = transfer_entropy(data, bins, lag)

    assert_allclose(py_result, cpp_result, rtol=RTOL, atol=ATOL)


def test_3d_te_matrix_properties(sample_3d_data, bins_3d):
    """Test properties of transfer entropy matrix."""
    data = sample_3d_data
    bins = bins_3d
    lag = 1

    for backend in [Backend.PY.value, Backend.CPP.value]:
        set_backend(backend)
        result_matrix = transfer_entropy(data, bins, lag)

        # Check matrix shape
        assert result_matrix.shape == (N_VARS, N_VARS)

        # Check diagonal is zero (no self-transfer entropy)
        assert_allclose(np.diag(result_matrix), np.zeros(N_VARS), rtol=RTOL, atol=ATOL)

        # Check non-negativity
        assert np.all(result_matrix >= -ATOL)


def test_univariate_entropy_consistency(sample_3d_data, bins_3d):
    """Test consistency between Python and CPP backends for univariate entropy."""
    data = sample_3d_data
    bins = bins_3d

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_results = []
    for i in range(N_VARS):
        py_results.append(entropy(data[:, i], bins))

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_results = []
    for i in range(N_VARS):
        cpp_results.append(entropy(data[:, i], bins))

    # Compare results for each variable
    for i in range(N_VARS):
        assert_allclose(py_results[i], cpp_results[i], rtol=RTOL, atol=ATOL)


def test_joint_entropy_consistency(sample_3d_data, bins_3d):
    """Test consistency between Python and CPP backends for joint entropy."""
    data = sample_3d_data
    bins = bins_3d

    # Test all possible pairs
    pairs = [(0, 1), (0, 2), (1, 2)]

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_matrix_results = joint_entropy(data, bins)
    py_pair_results = {}
    for pair in pairs:
        py_pair_results[pair] = joint_entropy(data, bins, at=pair)

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_matrix_results = joint_entropy(data, bins)
    cpp_pair_results = {}
    for pair in pairs:
        cpp_pair_results[pair] = joint_entropy(data, bins, at=pair)

    # Compare results for pairs
    for pair in pairs:
        assert_allclose(
            py_pair_results[pair], cpp_pair_results[pair], rtol=RTOL, atol=ATOL
        )

    # Compare matrix results
    assert_allclose(cpp_matrix_results, py_matrix_results, rtol=RTOL, atol=ATOL)

    # Verify that joint entropy is greater than or equal to individual entropies
    set_backend(Backend.CPP.value)  # Use any backend for this test
    for pair in pairs:
        h_joint = joint_entropy(data, bins, at=pair)
        h1 = entropy(data[:, pair[0]], bins)
        h2 = entropy(data[:, pair[1]], bins)
        assert h_joint >= h1 - ATOL
        assert h_joint >= h2 - ATOL


def test_conditional_entropy_consistency(sample_3d_data, bins_3d):
    """Test consistency between Python and CPP backends for conditional entropy."""
    data = sample_3d_data
    bins = bins_3d

    # Test all possible pairs for conditioning
    pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]

    # Test with Python backend
    set_backend(Backend.PY.value)
    py_results = {}
    for pair in pairs:
        py_results[pair] = conditional_entropy(data, bins, at=pair)

    # Test with CPP backend
    set_backend(Backend.CPP.value)
    cpp_results = {}
    for pair in pairs:
        cpp_results[pair] = conditional_entropy(data, bins, at=pair)

    # Compare results
    for pair in pairs:
        assert_allclose(py_results[pair], cpp_results[pair], rtol=RTOL, atol=ATOL)

        # Verify that conditional entropy satisfies H(X|Y) ≤ H(X)
        i, j = pair
        h_cond = conditional_entropy(data, bins, at=pair)
        h_x = entropy(data[:, i], bins)
        assert h_cond <= h_x + ATOL


def test_entropy_chain_rule_3d(sample_3d_data, bins_3d):
    """Test entropy chain rule with 3D data."""
    data = sample_3d_data
    bins = bins_3d

    for backend in [Backend.PY.value, Backend.CPP.value]:
        set_backend(backend)

        # Test H(X,Y) = H(X) + H(Y|X)
        h_xy = joint_entropy(data, bins, at=(0, 1))
        h_x = entropy(data[:, 0], bins)
        h_y_given_x = conditional_entropy(data, bins, at=(1, 0))
        assert_allclose(h_xy, h_x + h_y_given_x, rtol=RTOL, atol=ATOL)


def test_3d_edge_cases():
    """Test edge cases with 3D data."""
    # Test with constant data
    data = np.zeros((100, 3))
    bins = 3
    lag = 1

    for backend in [Backend.PY.value, Backend.CPP.value]:
        set_backend(backend)
        result = transfer_entropy(data, bins, lag)
        assert_allclose(result, np.zeros((3, 3)), rtol=RTOL, atol=ATOL)

    # Test with perfectly correlated data
    x = np.arange(100)
    data = np.column_stack([x, x, x])

    for backend in [Backend.PY.value, Backend.CPP.value]:
        set_backend(backend)
        result = transfer_entropy(data, bins, lag)
        # Diagonal should be zero
        assert_allclose(np.diag(result), np.zeros(3), rtol=RTOL, atol=ATOL)
