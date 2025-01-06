"""Tests for chaotic map implementations."""

import numpy as np
import pytest

from te_toolbox.systems.maps import BellowsMap, ExponentialMap, LogisticMap, TentMap


@pytest.fixture
def sample_data():
    """Generate sample data points for testing."""
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


def lyapunov_exponent(map_func, x0, n_iter=1000):
    """Approximate the Lyapunov exponent for a map."""
    x = x0
    exponent_sum = np.zeros_like(x)
    for _ in range(n_iter):
        derivative = map_func.derivative(x)
        exponent_sum += np.log(abs(derivative))
        x = map_func(x)
    return exponent_sum / n_iter


def test_tent_map(sample_data):
    """Test tent map implementation."""
    tent = TentMap(r=2)
    result = tent(sample_data)

    # Test specific known values
    assert np.isclose(tent(np.array([0.25])), 0.5)  # Below 0.5
    assert np.isclose(tent(np.array([0.75])), 0.5)  # Above 0.5

    # Test array input works
    assert result.shape == sample_data.shape

    # Test values are in [0,1]
    assert np.all(result >= 0) and np.all(result <= 1)

    # Test Lyapunov exponent (positive for chaotic behavior)
    lyapunov = lyapunov_exponent(tent, np.array([0.1]))
    assert lyapunov > 0  # Lyapunov exponent should be positive for chaos

    # Test fixed points for tent map
    fixed_points = [0, 2 / 3]  # Fixed points: x*1 = 0, x*2 = 2/3
    for fp in fixed_points:
        assert np.isclose(tent(np.array([fp])), fp)


def test_logistic_map(sample_data):
    """Test logistic map implementation."""
    logistic = LogisticMap(r=4.0)
    result = logistic(sample_data)

    # Test with r=4 which is chaotic
    assert np.isclose(logistic(np.array([0.5])), 1.0)

    # Test array input works
    assert result.shape == sample_data.shape

    # Test values are in [0,1] for initial values in [0,1]
    assert np.all(result >= 0) and np.all(result <= 1)

    # Test Lyapunov exponent (positive for chaotic behavior)
    lyapunov = lyapunov_exponent(logistic, np.array([0.1]))
    assert lyapunov > 0  # Lyapunov exponent should be positive for chaos

    # Test fixed points for logistic map
    fixed_points = [0, 0.75]  # Fixed points: x*1 = 0, x*2 = 0.75
    for fp in fixed_points:
        assert np.isclose(logistic(np.array([fp])), fp)


def test_bellows_map(sample_data):
    """Test bellows map implementation."""
    bellows = BellowsMap(r=2.0, b=6.0)
    result = bellows(sample_data)

    # Test array input works
    assert result.shape == sample_data.shape

    # Test specific fixed point
    fixed_point = np.array([0.8])
    assert np.isclose(bellows(fixed_point), fixed_point)

    # Test Lyapunov exponent (positive for chaotic behavior)
    lyapunov = lyapunov_exponent(bellows, np.array([0.1]))
    assert lyapunov > 0  # Lyapunov exponent should be positive for chaos


def test_exponential_map(sample_data):
    """Test exponential map implementation."""
    exp_map = ExponentialMap(r=4.0)
    result = exp_map(sample_data)

    # Test array input works
    assert result.shape == sample_data.shape

    # Test normalization (output should be in [0,1])
    assert np.all(result >= 0) and np.all(result <= 1)

    # Test endpoints
    assert np.isclose(exp_map(np.array([0.0])), 0.0)
    assert np.isclose(exp_map(np.array([1.0])), 1.0)

    # Test Lyapunov exponent (positive for chaotic behavior)
    lyapunov = lyapunov_exponent(exp_map, np.array([0.1]))
    assert lyapunov > 0  # Lyapunov exponent should be positive for chaos

    # Test fixed points for exponential map
    fixed_points = [0, 1]  # Fixed points: x*1 = 0, x*2 = 1
    for fp in fixed_points:
        assert np.isclose(exp_map(np.array([fp])), fp)


@pytest.mark.parametrize(
    "map_class,params",
    [
        (TentMap, {"r": 2.0}),
        (LogisticMap, {"r": 4.0}),
        (BellowsMap, {"r": 2.0, "b": 6.0}),
        (ExponentialMap, {"r": 4.0}),
    ],
)
def test_map_properties(map_class, params, sample_data):
    """Test general properties that should hold for all maps."""
    map_func = map_class(**params)
    result = map_func(sample_data)

    # Shape preservation
    assert result.shape == sample_data.shape

    # No NaN values
    assert not np.any(np.isnan(result))

    # No infinite values
    assert not np.any(np.isinf(result))
