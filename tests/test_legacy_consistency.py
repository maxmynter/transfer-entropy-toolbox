"""Consistency tests comparing new te_toolbox with original thesis implementation."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Import the old implementation
from numpy.testing import assert_array_almost_equal

import te_toolbox as tb

# Import legacy implementation
from tests.legacy_implementation import thesis_package as tp
from tests.legacy_implementation.utils import prepare_causal_dependent_data

# Test parameters
NUMERIC_TOLERANCE = 1e-10
MAX_EXAMPLES = 10


@pytest.mark.legacy
def test_basic_consistency():
    """Test consistency with fixed test case."""
    # Generate test data
    rng = np.random.default_rng(42)
    x = rng.random(size=1000)
    x, y = prepare_causal_dependent_data(x, lambda x: x + np.sin(x), noise=0.1)
    data = np.column_stack([x, y])
    bins = np.linspace(np.min(data), np.max(data), 10)

    # Test all entropy measures
    measures = [
        ("Entropy X", lambda d, b: (tp.entropy(d[:, 0], b), tb.entropy(d[:, 0], b))),
        ("Entropy Y", lambda d, b: (tp.entropy(d[:, 1], b), tb.entropy(d[:, 1], b))),
        (
            "Joint Entropy",
            lambda d, b: (tp.joint_entropy(d, b), tb.joint_entropy(d, b)),
        ),
        (
            "Conditional Entropy",
            lambda d, b: (tp.conditional_entropy(d, b), tb.conditional_entropy(d, b)),
        ),
        (
            "Transfer Entropy",
            lambda d, b: (
                tp.transfer_entropy(d, lag=1, bins=b),
                tb.transfer_entropy(d, lag=1, bins=b),
            ),
        ),
        (
            "Normalized TE",
            lambda d, b: (
                tp.normalised_transfer_entropy(d, lag=1, bins=b),
                tb.normalized_transfer_entropy(d, lag=1, bins=b),
            ),
        ),
        (
            "Log-N TE",
            lambda d, b: (
                tp.logN_normalised_transfer_entropy(d, lag=1, bins=b),
                tb.logn_normalized_transfer_entropy(d, lag=1, bins=b),
            ),
        ),
    ]

    for name, func in measures:
        old_result, new_result = func(data, bins)
        assert_array_almost_equal(
            old_result, new_result, decimal=10, err_msg=f"Inconsistency in {name}"
        )
        print(f"✓ {name} consistent")


@pytest.mark.legacy
@given(
    st.integers(min_value=100, max_value=1000),  # data size
    st.integers(min_value=5, max_value=20),  # number of bins
    st.floats(min_value=0.01, max_value=0.5),  # noise level
)
@settings(max_examples=MAX_EXAMPLES)
def test_property_based_consistency(n_samples, n_bins, noise):
    """Test consistency across different parameter combinations."""
    # Generate data
    rng = np.random.default_rng(42)
    x = rng.random(size=n_samples)
    x, y = prepare_causal_dependent_data(x, lambda x: x + np.sin(x), noise=noise)
    data = np.column_stack([x, y])
    bins = np.linspace(np.min(data), np.max(data), n_bins)

    # Test transfer entropy measures
    measures = [
        (
            "Transfer Entropy",
            lambda d, b: (
                tp.transfer_entropy(d, lag=1, bins=b),
                tb.transfer_entropy(d, lag=1, bins=b),
            ),
        ),
        (
            "Normalized TE",
            lambda d, b: (
                tp.normalised_transfer_entropy(d, lag=1, bins=b),
                tb.normalized_transfer_entropy(d, lag=1, bins=b),
            ),
        ),
        (
            "Log-N TE",
            lambda d, b: (
                tp.logN_normalised_transfer_entropy(d, lag=1, bins=b),
                tb.logn_normalized_transfer_entropy(d, lag=1, bins=b),
            ),
        ),
    ]

    for name, func in measures:
        old_result, new_result = func(data, bins)
        assert_array_almost_equal(
            old_result,
            new_result,
            decimal=8,
            err_msg=f"Inconsistency in {name} with parameters: "
            f"n_samples={n_samples}, n_bins={n_bins}, noise={noise}",
        )
