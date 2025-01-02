"""Consistency check with old M.Sc. Thesis code."""

import os
import sys

# Resolve old thesis functions for import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import thesis_package as tp
from utils import prepare_causal_dependent_data

import te_toolbox as tb

np.set_printoptions(threshold=np.inf, suppress=True)
SEED = 42

# Build causally dependent time series
rng = np.random.default_rng(seed=SEED)
x = rng.random(size=1000)
x, y = prepare_causal_dependent_data(x, lambda x: x + np.sin(x), noise=0.1)

data = np.column_stack([x, y])

bins = np.linspace(np.min(data), np.max(data), 10)

oent_x = tp.entropy(x, bins)
oent_y = tp.entropy(y, bins)

nent_x = tb.entropy(x, bins)
nent_y = tb.entropy(y, bins)

ojent = tp.joint_entropy(data, bins)
njent = tb.joint_entropy(data, bins)

ocent = tp.conditional_entropy(data, bins)
ncent = tb.conditional_entropy(data, bins)

otent = tp.transfer_entropy(data, lag=1, bins=bins)
ntent = tb.transfer_entropy(data, lag=1, bins=bins)

ontent = tp.normalised_transfer_entropy(data, lag=1, bins=bins)
nntent = tb.normalized_transfer_entropy(data, lag=1, bins=bins)

ologntent = tp.logN_normalised_transfer_entropy(data, lag=1, bins=bins)
nlogntent = tb.logn_normalized_transfer_entropy(data, lag=1, bins=bins)

if __name__ == "__main__":
    print("--- ENTROPY ---")
    print("Old")
    print(f"H(x)={oent_x}, H(y) ={oent_y}")
    print("\n")
    print("New")
    print(f"H(x)={nent_x}, H(y) ={nent_y}")
    print("\n\n")

    print("--- JOINT ENTROPY ---")
    print("Old H(x,y)")
    print(f"{ojent}")
    print("\n")
    print("New: H(x,y)")
    print(f"{njent}")
    print("\n\n")

    print("--- CONDITIONAL ENTROPY ---")
    print("Old H(x | y)")
    print(f"{ocent}")
    print("\n")
    print("New: H(x | y)")
    print(f"{ncent}")
    print("\n\n")

    print("--- TRANSFER ENTROPY ---")
    print("Old TE(x, y)")
    print(f"{otent}")
    print("\n")
    print("New: TE(x, y)")
    print(f"{ntent}")
    print("\n\n")

    print("--- NORMALIZED TRANSFER ENTROPY ---")
    print("Old NTE(x, y)")
    print(f"{ontent}")
    print("\n")
    print("New: NTE(x, y)")
    print(f"{nntent}")
    print("\n\n")

    print("--- LOG N NORMALIZED TRANSFER ENTROPY ---")
    print("Old log(N)TE(x, y)")
    print(f"{ologntent}")
    print("\n")
    print("New: log(N)TE(x, y)")
    print(f"{nlogntent}")
    print("\n\n")
