"""Tests for transfer entropy measures."""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_almost_equal

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)
from tests.conftest import (
    NORMALIZED_CAUSAL_THRESHOLD,
    NUMERIC_TOLERANCE,
    SIGNIFICANCE_THRESHOLD,
)

TEST_COUPLING_STRENGTH = 5
TEST_NOISE_LEVEL = 0.5


def generate_coupled_series(
    n_steps: int, coupling: float, noise_level: float
) -> npt.NDArray[np.float64]:
    """Generate coupled time series where y depends on x with lag 1.

    Args:
    ----
        n_steps: Length of time series
        coupling: Coupling strength between x and y
        noise_level: Amount of noise in y

    Returns:
    -------
        Array with shape (n_steps, 2) containing [x, y]

    """
    x = np.random.normal(0, 1, n_steps)
    y = np.zeros(n_steps)
    for t in range(1, n_steps):
        y[t] = coupling * x[t - 1] + noise_level * np.random.normal()
    return np.column_stack([x, y])


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
    assert np.all(nte >= 0 - NUMERIC_TOLERANCE)
    assert np.all(nte <= 1 + NUMERIC_TOLERANCE)


def test_logn_normalized_te():
    """Test logN normalized TE calculation."""
    data = np.random.normal(0, 1, (100, 2))
    te = transfer_entropy(data, bins=10, lag=1)
    log_te = logn_normalized_transfer_entropy(data, bins=10, lag=1)

    # Should be smaller than regular TE
    assert np.all(log_te <= te + 1e-10)


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
    st.integers(min_value=5, max_value=30),
)
def test_te_detects_causality(n_steps, n_bins):
    """Test if TE correctly identifies causal relationships."""
    data = generate_coupled_series(n_steps, TEST_COUPLING_STRENGTH, TEST_NOISE_LEVEL)
    te = transfer_entropy(data, bins=n_bins, lag=1)

    # TE should be higher in causal direction
    assert te[1, 0] > te[0, 1], "Failed to detect causality"
    # Difference should be significant
    assert te[1, 0] - te[0, 1] > SIGNIFICANCE_THRESHOLD, "TE difference too small"


@given(
    st.integers(min_value=1000, max_value=5000),
    st.integers(min_value=5, max_value=30),
)
def test_nte_detects_causality(n_steps, n_bins):
    """Test if normalized TE correctly identifies causal relationships."""
    data = generate_coupled_series(n_steps, TEST_COUPLING_STRENGTH, TEST_NOISE_LEVEL)
    nte = normalized_transfer_entropy(data, bins=n_bins, lag=1)

    # NTE should be higher in causal direction
    assert nte[1, 0] > nte[0, 1], "NTE failed to detect causality"

    # Difference should be significant
    assert nte[1, 0] - nte[0, 1] > SIGNIFICANCE_THRESHOLD, "NTE difference too small"
    # Should be properly normalized
    assert np.all(nte >= -NUMERIC_TOLERANCE), "NTE values below 0"
    assert np.all(nte <= 1 + NUMERIC_TOLERANCE), "NTE values above 1"


@given(
    st.integers(min_value=5000, max_value=5000),
    st.integers(min_value=5, max_value=15),
)
def test_norm_te_independent_signals(n_steps, n_bins):
    """Test TE behavior with independent signals."""
    x = np.random.normal(0, 1, n_steps)
    y = np.random.normal(0, 1, n_steps)
    data = np.column_stack([x, y])

    nte = normalized_transfer_entropy(data, bins=n_bins, lag=1)
    logn_nte = logn_normalized_transfer_entropy(data, bins=n_bins, lag=1)

    # NTE and log(N)-NTE should be small for independent signals
    # Note: Transfer entropy values are only meaningful when compared relatively
    # to each other. Without normalization, there is no universal threshold for
    # detecting causality, making this test impractical for unnormalized TE
    assert np.all(
        np.abs(nte) < NORMALIZED_CAUSAL_THRESHOLD
    ), "NTE detected false causality"

    assert np.all(
        np.abs(logn_nte) < NORMALIZED_CAUSAL_THRESHOLD
    ), "log(N)-NTE detected false causality"


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


@pytest.mark.parametrize(
    "bins,expected_shape",
    [
        ([10, 10], (2, 2)),
        (7, (2, 2)),
        ([np.linspace(-3, 3, 11), np.linspace(-3, 3, 11)], (2, 2)),
        ([10, np.linspace(-3, 3, 11)], (2, 2)),
    ],
)
def test_logn_te_list_bins(bins, expected_shape):
    """Test logN normalized TE with list of bin specifications."""
    data = np.clip(generate_coupled_series(1000, 0.5, 0.1), -3, 3)
    result = logn_normalized_transfer_entropy(data, bins=bins, lag=1)
    assert result.shape == expected_shape
    assert np.all(result >= 0 - NUMERIC_TOLERANCE)
