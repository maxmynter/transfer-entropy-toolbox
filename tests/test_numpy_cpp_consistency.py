"""Testing the consistency between numpy and CPP implementations of entropies."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from te_toolbox.entropies.bivariate import conditional_entropy as ce
from te_toolbox.entropies.bivariate import discrete_joint_entropy as dje
from te_toolbox.entropies.multivariates import discrete_multivar_joint_entropy as dmje
from te_toolbox.entropies.transfer.base import discrete_transfer_entropy as dte
from te_toolbox.entropies.univariate import discrete_entropy as de
from te_toolbox.fast_entropy import discrete_conditional_entropy as fdce
from te_toolbox.fast_entropy import discrete_entropy as fde
from te_toolbox.fast_entropy import discrete_joint_entropy as fdje
from te_toolbox.fast_entropy import discrete_multivar_joint_entropy as fdmje
from te_toolbox.fast_entropy import discrete_transfer_entropy as fdte

# Constants for fixed test cases
N_SAMPLES = 1000
N_VARS = 3
RTOL = 1e-5  # Increased tolerance for float comparisons
ATOL = 1e-5  # Increased tolerance for float comparisons

# Additional tolerances for probabilistic tests
PROB_RTOL = 1e-4  # Relative tolerance for probabilistic tests
PROB_ATOL = 1e-4  # Absolute tolerance for probabilistic tests


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randint(0, N_VARS, size=N_SAMPLES)  # noqa: N806 # X for a dataset
    Y = X + np.random.randint(0, N_VARS, size=N_SAMPLES)  # noqa: N806 # Y for a dataset
    return X, Y, np.column_stack([X, Y])


@pytest.fixture
def bins():
    """Generate bin edges for discretization."""
    bins_x = np.array([-0.5, 0.5, 1.5, 2.5])
    bins_y = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    return bins_x, bins_y


def test_discrete_entropy(sample_data):
    """Test consistency between legacy and fast discrete entropy."""
    X, _, _ = sample_data  # noqa: N806 # X for a dataset
    unique_x = len(np.unique(X))

    legacy_result = de(X, unique_x)
    fast_result = fde(X, unique_x)

    assert_allclose(legacy_result, fast_result, rtol=RTOL, atol=ATOL)


@given(st.integers(min_value=100, max_value=10000))
def test_discrete_entropy_hypothesis(n_samples):
    """Property-based test for discrete entropy consistency."""
    X = np.random.randint(0, N_VARS, size=n_samples)  # noqa: N806 # X for a dataset
    unique_x = len(np.unique(X))

    legacy_result = de(X, unique_x)
    fast_result = fde(X, unique_x)

    assert_allclose(legacy_result, fast_result, rtol=RTOL, atol=ATOL)


def test_joint_entropy(sample_data):
    """Test consistency between legacy and fast joint entropy."""
    _, _, data2d = sample_data
    unique_vals = [N_VARS, N_VARS * 2 - 1]

    legacy_result = dje(data2d, unique_vals, at=(0, 1))
    fast_result = fdje(data2d, unique_vals)

    assert_allclose(legacy_result, fast_result, rtol=RTOL, atol=ATOL)


def test_multivariate_joint_entropy(sample_data):
    """Test consistency between legacy and fast multivariate joint entropy."""
    X, Y, _ = sample_data  # noqa: N806 # X, Y for a dataset
    unique_vals = [N_VARS, N_VARS * 2 - 1]

    legacy_result = dmje([X, Y], unique_vals)
    fast_result = fdmje([X, Y], unique_vals)

    assert_allclose(legacy_result, fast_result, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_transfer_entropy(sample_data, lag):
    """Test consistency between legacy and fast transfer entropy."""
    _, _, data2d = sample_data
    unique_vals = [N_VARS, N_VARS * 2 - 1]

    legacy_result = dte(data2d, n_classes=unique_vals, lag=lag, at=(0, 1))
    fast_result = fdte(data2d, n_classes=unique_vals, lag=lag)

    assert_allclose(legacy_result, fast_result, rtol=RTOL, atol=ATOL)


def test_conditional_entropy_with_discretization(sample_data, bins):
    """Test consistency between numpy and CPP implementation."""
    _, _, data2d = sample_data
    bins_x, bins_y = bins
    unique_vals = [N_VARS, N_VARS * 2 - 1]

    # Legacy implementation requires discretized data
    legacy_result = ce(data2d, [bins_x, bins_y], at=(0, 1))

    # Fast implementation works with original data
    fast_result = fdce(data2d, unique_vals)

    # Note: Results might differ slightly due to discretization
    assert_allclose(legacy_result, fast_result, rtol=1e-2, atol=1e-2)


def test_entropy_chain_rule():
    """Test entropy chain rule with controlled data for better numerical stability."""
    # Use fixed data with known properties
    n_samples = N_SAMPLES * 100  # Larger sample size for better convergence
    np.random.seed(42)

    # Create independent variables with known distributions
    p_x = np.array([0.3, 0.7])  # Probability distribution for X
    p_y = np.array([0.4, 0.6])  # Probability distribution for Y

    # Generate samples
    X = np.random.choice(2, size=n_samples, p=p_x)  # noqa: N806 # X for a dataset
    Y = np.random.choice(2, size=n_samples, p=p_y)  # noqa: N806 # Y for a dataset
    data2d = np.column_stack([X, Y])

    # Calculate entropies
    h_x = fde(X, 2)
    h_y = fde(Y, 2)
    h_xy = fdje(data2d, [2, 2])
    h_y_given_x = fdce(data2d, [2, 2])

    # Calculate empirical probabilities
    p_x_empirical = np.bincount(X) / len(X)
    p_y_empirical = np.bincount(Y) / len(Y)

    # Theoretical values using empirical probabilities
    h_x_theory = -np.sum(p_x_empirical * np.log(p_x_empirical))
    h_y_theory = -np.sum(p_y_empirical * np.log(p_y_empirical))

    # Test empirical values against theoretical values
    assert_allclose(h_x, h_x_theory, rtol=PROB_RTOL, atol=PROB_ATOL)
    assert_allclose(h_y, h_y_theory, rtol=PROB_RTOL, atol=PROB_ATOL)

    # Test entropy chain rule
    assert_allclose(h_xy, h_x + h_y, rtol=PROB_RTOL, atol=PROB_ATOL)
    assert h_y_given_x <= h_y + PROB_ATOL


@given(
    st.integers(min_value=2, max_value=5),  # n_vars
    st.integers(min_value=1000, max_value=2000),  # n_samples
)
def test_entropy_general_properties(n_vars, n_samples):
    """Test general entropy properties with hypothesis testing."""
    np.random.seed(42)  # Fixed seed for reproducibility

    # Generate independent variables
    X = np.random.randint(0, n_vars, size=n_samples)  # noqa: N806 # X for a dataset
    Y = np.random.randint(0, n_vars, size=n_samples)  # noqa: N806 # Y for a dataset
    data2d = np.column_stack([X, Y])
    unique_vals = [n_vars, n_vars]

    # Calculate entropies
    h_x = fde(X, n_vars)
    h_y = fde(Y, n_vars)
    h_xy = fdje(data2d, unique_vals)

    # Test basic properties
    assert h_x >= 0  # Non-negativity
    assert h_y >= 0
    assert h_xy >= 0

    # Upper bounds
    assert h_x <= np.log(n_vars) + PROB_ATOL
    assert h_y <= np.log(n_vars) + PROB_ATOL
    assert h_xy <= np.log(n_vars * n_vars) + PROB_ATOL


def test_edge_cases():
    """Test entropy calculations with edge cases."""
    # Test with constant array (should give 0 entropy)
    X = np.ones(100, dtype=int)  # noqa: N806 # X for a dataset
    assert_allclose(fde(X, 1), 0.0, atol=ATOL)

    # Test with uniform distribution
    n = 1000
    n_classes = 4
    X = np.random.randint(0, n_classes, size=n)  # noqa: N806 # X for a dataset
    # Theoretical maximum entropy for uniform distribution
    max_entropy = np.log(n_classes)
    # Allow some deviation due to finite sampling
    assert fde(X, n_classes) <= max_entropy + ATOL

    # Test independence property with controlled data
    X = np.array([0, 1] * 50)  # noqa: N806 # X for a dataset
    Y = np.array([0, 0, 1, 1] * 25)  # noqa: N806 # Y for a dataset
    data2d = np.column_stack([X, Y])

    h_x = fde(X, 2)
    h_y = fde(Y, 2)
    h_xy = fdje(data2d, [2, 2])

    # For independent variables, joint entropy should be sum of individual entropies
    assert_allclose(h_xy, h_x + h_y, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
