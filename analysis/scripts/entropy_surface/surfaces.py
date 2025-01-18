"""Generate TE by sample size and binning detail surface plots."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surfaces_constant import (
    LAG,
    N_ITER,
    N_MAPS,
    N_TRANSIENT,
    SEED,
    SURFACE_PLOT_DIR,
    bin_range,
    length_range,
    maps,
)

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)
from te_toolbox.preprocessing import remap_to
from te_toolbox.systems.lattice import CMLConfig, CoupledMapLatticeGenerator

# Configure plotting
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stix",
    }
)
sns.set_style("whitegrid")


def avg_off_diag(matrix: np.ndarray):
    """Average off-diagonal elements of a 2D numpy array."""
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.mean(matrix[mask])


def generate_data(map_func):
    """Generate CML data using constants."""
    config = CMLConfig(
        map_function=map_func,
        n_maps=N_MAPS,
        coupling_strength=0.5,
        n_steps=N_ITER,
        warmup_steps=N_TRANSIENT,
        seed=SEED,
    )
    return CoupledMapLatticeGenerator(config).generate().lattice


def plot_measure_surface(
    measure_vals: np.ndarray, bins: np.ndarray, lengths: np.ndarray, measure_name: str
):
    """Plot measure surface using trisurf."""
    x_vals, y_vals = np.meshgrid(bins, lengths)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x_vals,
        y_vals,
        measure_vals,
        vmin=0,
        cmap="viridis",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("Number of bins")
    ax.set_ylabel("Sample length")
    ax.set_zlabel(measure_name, rotation=90)

    plt.tight_layout()

    plt.savefig(
        SURFACE_PLOT_DIR / f"{measure_name.lower()}_surface.png",
        dpi=300,
    )
    plt.close()


def main():
    """Generate sample-bin-te dependence surfaces."""
    data = generate_data(maps["TentMap(r=2)"])

    rng = np.random.default_rng(SEED)
    gaussian_samples = rng.normal(size=data.shape)
    data = remap_to(data, gaussian_samples, rng)

    surfaces = {
        name: np.zeros((len(length_range), len(bin_range)))
        for name in ["TE", "NTE", "logNTE"]
    }

    for i, length in enumerate(length_range):
        print(f"Processing length {length}")
        data_subset = data[:length]

        for j, n_bins in enumerate(bin_range):
            bins = np.linspace(np.min(data_subset), np.max(data_subset), n_bins + 1)

            te = avg_off_diag(transfer_entropy(data_subset, bins, LAG))
            nte = avg_off_diag(normalized_transfer_entropy(data_subset, bins, LAG))
            lognte = avg_off_diag(
                logn_normalized_transfer_entropy(data_subset, bins, LAG)
            )

            surfaces["TE"][i, j] = te
            surfaces["NTE"][i, j] = nte
            surfaces["logNTE"][i, j] = lognte

    # Create plots
    for name, surface in surfaces.items():
        plot_measure_surface(surface, bin_range, length_range, name)


if __name__ == "__main__":
    main()
