"""Benchmarking of entropy calculation performance."""

import timeit

import numpy as np

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)

# Generate sample data
n_samples = 10000
n_vars = 3
data = np.random.randn(n_samples, n_vars)
bins = 10
lag = 1


# Benchmark different functions
def benchmark(n=100):
    """Run a benchmark to get execution time of transfer entropy functions."""
    # Run once pre timing to catch numba compilation
    transfer_entropy(data, bins, lag)
    normalized_transfer_entropy(data, bins, lag)
    logn_normalized_transfer_entropy(data, bins, lag)

    # Timing of execution

    # Basic transfer entropy
    te_time = timeit.timeit(lambda: transfer_entropy(data, bins, lag), number=n)

    # Normalized transfer entropy
    nte_time = timeit.timeit(
        lambda: normalized_transfer_entropy(data, bins, lag), number=n
    )

    # Log-normalized transfer entropy
    lnte_time = timeit.timeit(
        lambda: logn_normalized_transfer_entropy(data, bins, lag), number=n
    )

    print(f"Transfer Entropy: {te_time:.4f}s, per call {te_time / n:.4f}s")
    print(f"Normalized TE: {nte_time:.4f}s, per call {nte_time / n:.4f}s")
    print(f"Log-normalized TE: {lnte_time:.4f}s, per call {lnte_time / n:.4f}s")


if __name__ == "__main__":
    benchmark()
