"""Benchmark Entropy Numpy and CPP implementation."""

import timeit

import numpy as np

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

n_samples = 10**5
n_vars1 = 3
n_vars2 = 3

X = np.random.randint(0, n_vars1, size=n_samples)
Y = X + np.random.randint(0, n_vars2, size=n_samples)
data2d = np.column_stack([X, Y])

UNIQUE_X = n_vars1
UNIQUE_Y = n_vars1 + n_vars2 - 1

BINS_X = np.array([-0.5, 0.5, 1.5, 2.5])
BINS_Y = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])


def benchmark(title, python, cpp, n=100):
    """Benchmark fast entropies."""
    # Call the functions such that numba JIT is warmed up
    print()
    print(f"=========  {title}  ========")
    print("Python Value: ", python())
    print("CPP Value: ", cpp())

    pyte = timeit.timeit(python, number=n)
    cppte = timeit.timeit(cpp, number=n)

    print()
    print(f"Python Entropy: {pyte:.4f}s, per call {pyte / n:.4f}s")
    print(f"CPP Entropy: {cppte:.4f}s, per call {cppte / n:.4f}s")
    print()


if __name__ == "__main__":
    benchmark("Entropy", lambda: de(X, UNIQUE_X), lambda: fde(X, UNIQUE_X))

    benchmark(
        "Joint Entropy",
        lambda: dje(data2d, [UNIQUE_X, UNIQUE_Y], at=(0, 1)),
        lambda: fdje(data2d, [UNIQUE_X, UNIQUE_Y]),
    )

    benchmark(
        "Multivariate Joint Entropy",
        lambda: dmje([X, Y], [UNIQUE_X, UNIQUE_Y]),
        lambda: fdmje([X, Y], [UNIQUE_X, UNIQUE_Y]),
    )
    benchmark(
        "Conditional Entropy",
        lambda: ce(data2d, [BINS_X, BINS_Y], at=(0, 1)),  # H(Y|X)
        lambda: fdce(data2d, [UNIQUE_X, UNIQUE_Y]),  # H(Y|X)
    )
    benchmark(
        "Transfer Entropy",
        lambda: dte(
            data2d, n_classes=[UNIQUE_X, UNIQUE_Y], lag=1, at=(0, 1)
        ),  # TE Y->X
        lambda: fdte(data2d, n_classes=[UNIQUE_X, UNIQUE_Y], lag=1),  # TE Y->X
    )
