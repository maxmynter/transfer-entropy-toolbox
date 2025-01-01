"""Tests for transfer entropy measures."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    prepare_te_data,
    transfer_entropy,
)
from tests.conftest import (
    NORMALIZED_CAUSAL_THRESHOLD,
    NUMERIC_TOLERANCE,
    SIGNIFICANCE_THRESHOLD,
    bin_generator,
    regularize_hypothesis_generated_data,
)


def generate_coupled_series(
    n_steps: int, coupling: float, noise_level: float
) -> npt.NDArray[np.float64]:
    """Generate coupled time series where y depends on x with lag 1.

    Args:
        n_steps: Length of time series
        coupling: Coupling strength between x and y
        noise_level: Amount of noise in y

    Returns:
        Array with shape (n_steps, 2) containing [x, y]

    """
    x = np.random.normal(0, 1, n_steps)
    y = np.zeros(n_steps)
    for t in range(1, n_steps):
        y[t] = coupling * x[t - 1] + noise_level * np.random.normal()
    return np.column_stack([x, y])


def test_prepare_te_data():
    """Test data preparation for TE calculation."""
    data = np.array([[1, 4], [2, 5], [3, 6]])
    lag = 1
    bins = 3

    current, lagged, bin_list = prepare_te_data(data, lag, bins)
    assert current.shape[0] == lagged.shape[0]
    assert current.shape[1] == lagged.shape[1]
    assert len(bin_list) == data.shape[1]

    with pytest.raises(ValueError):
        prepare_te_data(np.array([1, 2, 3]), lag, bins)


def test_transfer_entropy_zero():
    """Test TE calculation for independent processes."""
    # Two independent random walks
    n_steps = 1000
    x = np.cumsum(np.random.normal(0, 1, n_steps))
    y = np.cumsum(np.random.normal(0, 1, n_steps))
    data = np.column_stack([x, y])

    te = transfer_entropy(data, bins=10, lag=1)
    # TE should be close to zero for independent processes
    assert_array_almost_equal(te, np.zeros_like(te), decimal=1)


def test_transfer_entropy_causality():
    """Test TE calculation for coupled processes."""
    n_steps = 1000
    x = np.random.normal(0, 1, n_steps)
    y = np.zeros(n_steps)
    # y depends on x with lag 1
    for t in range(1, n_steps):
        y[t] = 0.5 * x[t - 1] + 0.1 * np.random.normal()

    data = np.column_stack([x, y])
    te = transfer_entropy(data, bins=10, lag=1)
    # TE x->y should be larger than y->x
    assert te[1, 0] > te[0, 1]


def test_normalized_te_bounds():
    """Test normalized TE is bounded between 0 and 1."""
    n_steps = 1000
    x = np.random.normal(0, 1, n_steps)
    y = np.random.normal(0, 1, n_steps)
    data = np.column_stack([x, y])

    nte = normalized_transfer_entropy(data, bins=10, lag=1)
    assert np.all(nte >= 0)
    assert np.all(nte <= 1)


def test_logn_normalized_te():
    """Test logN normalized TE calculation."""
    data = np.random.normal(0, 1, (100, 2))
    te = transfer_entropy(data, bins=10, lag=1)
    log_te = logn_normalized_transfer_entropy(data, bins=10, lag=1)

    # Should be smaller than regular TE
    assert np.all(log_te <= te + 1e-10)


@given(
    st.lists(st.floats(min_value=-100, max_value=100), min_size=50, unique=True),
    st.lists(st.floats(min_value=-100, max_value=100), min_size=50, unique=True),
)
def test_te_symmetry(x, y):
    """Test TE symmetry properties."""
    data = regularize_hypothesis_generated_data(x, y)
    bins = bin_generator(data, 10)
    te = transfer_entropy(data, bins, lag=1)

    # Self-transfer entropy should be equal
    assert_almost_equal(te[0, 0], te[1, 1])


@given(st.integers(min_value=1, max_value=10))
def test_lag_dependency(lag):
    """Test TE behavior with different lags."""
    n_steps = 1000
    x = np.random.normal(0, 1, n_steps)
    y = np.zeros(n_steps)
    # y depends on x with specific lag
    for t in range(lag, n_steps):
        y[t] = 0.5 * x[t - lag] + 0.1 * np.random.normal()

    data = np.column_stack([x, y])
    te = transfer_entropy(data, bins=10, lag=lag)
    # TE should detect the correct lag dependency
    assert te[1, 0] > te[0, 1]


@given(
    st.integers(min_value=1000, max_value=5000),
    st.floats(min_value=0.1, max_value=0.9),
    st.floats(min_value=0.01, max_value=0.2),
    st.integers(min_value=5, max_value=30),
)
def test_te_detects_causality(n_steps, coupling, noise_level, n_bins):
    """Test if TE correctly identifies causal relationships."""
    data = generate_coupled_series(n_steps, coupling, noise_level)
    te = transfer_entropy(data, bins=n_bins, lag=1)

    # TE should be higher in causal direction
    assert te[1, 0] > te[0, 1], "Failed to detect causality"
    # Difference should be significant
    assert te[1, 0] - te[0, 1] > SIGNIFICANCE_THRESHOLD, "TE difference too small"


@given(
    st.integers(min_value=1000, max_value=5000),
    st.floats(min_value=0.5, max_value=0.9),
    st.floats(min_value=0.01, max_value=0.2),
    st.integers(min_value=5, max_value=30),
)
def test_nte_detects_causality(n_steps, coupling, noise_level, n_bins):
    """Test if normalized TE correctly identifies causal relationships."""
    data = generate_coupled_series(n_steps, coupling, noise_level)
    nte = normalized_transfer_entropy(data, bins=n_bins, lag=1)

    # NTE should be higher in causal direction
    assert nte[1, 0] > nte[0, 1], "NTE failed to detect causality"

    # Difference should be significant
    assert nte[1, 0] - nte[0, 1] > SIGNIFICANCE_THRESHOLD, "NTE difference too small"
    # Should be properly normalized
    assert np.all(nte >= -NUMERIC_TOLERANCE), "NTE values below 0"
    assert np.all(nte <= 1 + NUMERIC_TOLERANCE), "NTE values above 1"


@given(
    st.integers(min_value=1000, max_value=5000),
    st.integers(min_value=5, max_value=30),
)
def test_te_independent_signals(n_steps, n_bins):
    """Test TE behavior with independent signals."""
    x = np.random.normal(0, 1, n_steps)
    y = np.random.normal(0, 1, n_steps)
    data = np.column_stack([x, y])

    te = transfer_entropy(data, bins=n_bins, lag=1)
    nte = normalized_transfer_entropy(data, bins=n_bins, lag=1)

    # TE and NTE should be small for independent signals
    assert np.all(np.abs(te) < NORMALIZED_CAUSAL_THRESHOLD), "Detected false causality"
    assert np.all(
        np.abs(nte) < NORMALIZED_CAUSAL_THRESHOLD
    ), "NTE detected false causality"


@given(
    st.integers(min_value=1000, max_value=5000),
    st.floats(min_value=0.5, max_value=0.9),
    st.floats(min_value=0.01, max_value=0.2),
    st.integers(min_value=5, max_value=30),
)
def test_log_normalized_te(n_steps, coupling, noise_level, n_bins):
    """Test if log-normalized TE preserves causality detection."""
    data = generate_coupled_series(n_steps, coupling, noise_level)

    te = transfer_entropy(data, bins=n_bins, lag=1)
    log_te = logn_normalized_transfer_entropy(data, bins=n_bins, lag=1)

    # Log-normalized TE should be smaller than regular TE
    assert np.all(log_te <= te + 1e-10)
    # Should preserve directionality
    assert log_te[1, 0] > log_te[0, 1], "Log-normalized TE failed to detect causality"
