"""Benchmark Entropy Numpy and CPP implementation."""

import timeit

import numpy as np

from te_toolbox.entropies import discrete_entropy as de
from te_toolbox.fast_entropy import discrete_entropy as fde

n_samples = 10**5
n_vars = 3
data = np.random.randint(0, n_vars, size=n_samples)


def benchmark(n=100):
    """Benchmark fast n slow entropies."""
    print(de(data, n_vars))

    print(fde(data, n_vars))
    de_time = timeit.timeit(lambda: de(data, n_vars), number=n)
    fde_time = timeit.timeit(lambda: fde(data, n_vars), number=n)

    print(f"Python Entropy: {de_time:.4f}s, per call {de_time / n:.4f}s")
    print(f"CPP Entropy: {fde_time:.4f}s, per call {fde_time / n:.4f}s")


if __name__ == "__main__":
    benchmark()
