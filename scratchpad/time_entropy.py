"""Benchmark Entropy Numpy and CPP implementation."""

import timeit

import numpy as np

from te_toolbox.entropies.multivariates import discrete_multivar_joint_entropy as dmje
from te_toolbox.entropies.univariate import discrete_entropy as de
from te_toolbox.fast_entropy import discrete_entropy as fde
from te_toolbox.fast_entropy import discrete_multivar_joint_entropy as fdmje

n_samples = 10**5
n_vars = 3
data = np.random.randint(0, n_vars, size=n_samples)
data2 = np.random.randint(0, n_vars, size=n_samples)


def benchmark(title, python, cpp, n=100):
    """Benchmark fast entropies."""
    # Call the functions such that numba JIT is warmed up
    print("Python Value: ", python())
    print("CPP Value: ", cpp())

    pyte = timeit.timeit(python, number=n)
    cppte = timeit.timeit(cpp, number=n)

    print(f"=========  {title}  ========")
    print(f"Python Entropy: {pyte:.4f}s, per call {pyte / n:.4f}s")
    print(f"CPP Entropy: {cppte:.4f}s, per call {cppte / n:.4f}s")
    print()


if __name__ == "__main__":
    benchmark("Entropy", lambda: de(data, n_vars), lambda: fde(data, n_vars))

    benchmark(
        "Multivariate Joint Entropy",
        lambda: dmje([data, data2], [n_vars] * 2),
        lambda: fdmje([data, data2], [n_vars] * 2),
    )
