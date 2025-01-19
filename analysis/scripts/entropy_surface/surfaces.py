"""Generate TE by sample size and binning detail surface plots."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surfaces_constant import (
    EPS,
    LAG,
    N_BINS,
    N_ITER,
    N_LENS,
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


def avg_upper_tri(matrix: np.ndarray):
    """Average upper triangular (left-to-right) elements of a 2D numpy array."""
    upper_tri = np.triu_indices(matrix.shape[0], k=1)
    return np.mean(matrix[upper_tri])


def generate_data(map_func):
    """Generate CML data using constants."""
    config = CMLConfig(
        map_function=map_func,
        n_maps=N_MAPS,
        coupling_strength=EPS,
        n_steps=N_ITER,
        warmup_steps=N_TRANSIENT,
        seed=SEED,
    )
    return CoupledMapLatticeGenerator(config).generate().lattice


def compute_surfaces(data: np.ndarray) -> dict:
    """Compute TE surfaces for different measures."""
    surfaces = {
        name: np.zeros((len(length_range), len(bin_range)))
        for name in ["TE", "NTE", "logNTE"]
    }

    for i, length in enumerate(length_range):
        print(f"Processing length {length}")
        data_subset = data[:length]
        for j, n_bins in enumerate(bin_range):
            bins = np.linspace(np.min(data_subset), np.max(data_subset), n_bins + 1)
            te = avg_upper_tri(transfer_entropy(data_subset, bins, LAG))
            nte = avg_upper_tri(normalized_transfer_entropy(data_subset, bins, LAG))
            lognte = avg_upper_tri(
                logn_normalized_transfer_entropy(data_subset, bins, LAG)
            )
            surfaces["TE"][i, j] = te
            surfaces["NTE"][i, j] = nte
            surfaces["logNTE"][i, j] = lognte

    return surfaces


def save_surfaces(surfaces: dict, filename: str):
    """Save computed surfaces to a pickle file."""
    save_path = SURFACE_PLOT_DIR / filename
    with open(save_path, "wb") as f:
        pickle.dump(surfaces, f)


def load_surfaces(filename: str) -> dict:
    """Load computed surfaces from a pickle file."""
    load_path = SURFACE_PLOT_DIR / filename
    with open(load_path, "rb") as f:
        return pickle.load(f)


def plot_measure_surface(
    measure_vals: np.ndarray,
    bins: np.ndarray,
    lengths: np.ndarray,
    measure_name: str,
    **plot_kwargs,
):
    """Plot measure surface using trisurf with customizable plot parameters."""
    x_vals, y_vals = np.meshgrid(bins, lengths)
    fig = plt.figure(figsize=plot_kwargs.get("figsize", (10, 8)))
    ax = fig.add_subplot(111, projection="3d")

    plot_params = {
        "vmin": 0,
        "cmap": "viridis",
        "alpha": 0.9,
        "linewidth": 0,
        "antialiased": True,
    }
    plot_params.update(plot_kwargs.get("surface_params", {}))

    surface = ax.plot_surface(x_vals, y_vals, measure_vals, **plot_params)

    ax.set_xlabel(plot_kwargs.get("xlabel", "Number of bins"))
    ax.set_ylabel(plot_kwargs.get("ylabel", "Sample length"))
    ax.set_zlabel(plot_kwargs.get("zlabel", measure_name), rotation=90)

    if plot_kwargs.get("add_colorbar", True):
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    filename = plot_kwargs.get(
        "filename",
        f"{measure_name.lower()}x{N_MAPS}_surface_{N_LENS}data_{N_BINS}bins.png",
    )
    plt.savefig(SURFACE_PLOT_DIR / filename, dpi=plot_kwargs.get("dpi", 300))
    plt.close()


def compute_and_save_surfaces(map_name, savename):
    """Compute and save surfaces to file."""
    data = generate_data(maps[map_name])
    surfaces = compute_surfaces(data)
    save_surfaces(surfaces, savename)
    return surfaces


def plot_all_surfaces(surfaces, plot_kwargs):
    """Plot all surfaces with given plot parameters."""
    if plot_kwargs is None:
        plot_kwargs = {}

    for name, surface in surfaces.items():
        plot_measure_surface(surface, bin_range, length_range, name, **plot_kwargs)


def main():
    """Scan (N)TE for samples and bin."""
    for map_name, _ in maps.items():
        print("Evaluating map", map_name)
        filename = (
            f"surfaces_{map_name}_{EPS}eps_x{N_MAPS}"
            f"_surface_{N_LENS}data_{N_BINS}bins.pkl"
        )
        # Check if saved surfaces exist
        if not (SURFACE_PLOT_DIR / filename).exists():
            print("Computing and saving surfaces...")
            surfaces = compute_and_save_surfaces(map_name, filename)
        else:
            print("Loading pre-computed surfaces...")
            surfaces = load_surfaces(filename)

        # Plot surfaces with default parameters
        plot_all_surfaces(surfaces)


if __name__ == "__main__":
    main()
