"""Plot TE and it's std for fixed bin number and increasing sample size."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from te_toolbox.entropies import logn_normalized_transfer_entropy, transfer_entropy
from te_toolbox.entropies.transfer.normalized import normalized_transfer_entropy
from te_toolbox.systems import CMLConfig
from te_toolbox.systems.lattice import CoupledMapLatticeGenerator
from te_toolbox.systems.maps import TentMap

N_MAPS = 100
bins = np.linspace(0, 1, 10)

ents = {
    "TE": transfer_entropy,
    "logNTE": logn_normalized_transfer_entropy,
    "NTE": normalized_transfer_entropy,
}
savepath = Path("analysis/plots/te_fixed_bins/")
savepath.mkdir(exist_ok=True)

sns.set()
plt.figure()

for te_name, te_func in ents.items():
    print(f"Measure: {te_name}")
    te = []
    te_std = []

    sizes = [int(s) for s in np.geomspace(50, 1.5 * 10**4, 100)]

    for size in sizes:
        print(f"Size: {size}")
        config = CMLConfig(
            map_function=TentMap(r=2),
            n_maps=N_MAPS,
            coupling_strength=0.5,
            n_steps=size,
            warmup_steps=10**5,
            seed=42,
        )
        cml = CoupledMapLatticeGenerator(config).generate().lattice
        te_s = []
        for i in range(1, N_MAPS):
            te_s.append(te_func(data=cml[:, [i - 1, i]], bins=bins, lag=1, at=(1, 0)))

        te.append(np.mean(te_s))
        te_std.append(np.std(te_s))
    plt.errorbar(sizes, te, yerr=te_std, label=te_name)
    plt.xscale("log")
    plt.xlabel("Sample Size")
    plt.ylabel("(N)TE")
plt.legend()
plt.tight_layout()
plt.savefig(savepath / f"{te_name}_fixed_bins_by_sample_size.png", dpi=300)
