"""Plot TE and it's std for fixed bin number and increasing sample size."""

import matplotlib.pyplot as plt
import numpy as np

from te_toolbox.entropies import transfer_entropy
from te_toolbox.systems import CMLConfig
from te_toolbox.systems.lattice import CoupledMapLatticeGenerator
from te_toolbox.systems.maps import TentMap

N_MAPS = 100
bins = np.linspace(0, 1, 10)

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
        te_s.append(
            transfer_entropy(data=cml[:, [i - 1, i]], bins=bins, lag=1, at=(1, 0))
        )

    te.append(np.mean(te_s))
    te_std.append(np.std(te_s))
plt.errorbar(sizes, te, yerr=te_std)
plt.xscale("log")
plt.xlabel("Sample Size")
plt.ylabel("TE")
plt.savefig("te_fixed_bins_by_sample_size.png", dpi=300)
