"""Generate TE by sample size and binning detail surface plots."""

import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from surfaces_constant import (
    EPS,
    LAG,
    N_BINS,
    N_ITER,
    N_LENS,
    N_MAPS,
    N_TRANSIENT,
    SEED,
    SURFACE_DATA_DIR,
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

NORM_STD = "sig_by_mu"
MEAN = "mean"


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


def compute_te_for(data, length, n_bins):
    """Compute TE values for length x bins combination."""
    data_subset = data[: int(length)]
    bins = np.linspace(np.min(data_subset), np.max(data_subset), n_bins + 1)

    te_vals = []
    nte_vals = []
    lognte_vals = []

    for k in range(N_MAPS - 1):
        pair_data = data_subset[:, [k, k + 1]]
        te_vals.append(transfer_entropy(pair_data, bins, LAG, at=(1, 0)))
        nte_vals.append(normalized_transfer_entropy(pair_data, bins, LAG, at=(1, 0)))
        lognte_vals.append(
            logn_normalized_transfer_entropy(pair_data, bins, LAG, at=(1, 0))
        )

    results = {}
    for vals, keys in [(te_vals, "TE"), (nte_vals, "NTE"), (lognte_vals, "logNTE")]:
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        results[keys] = {
            MEAN: mean_val,
            NORM_STD: std_val / mean_val if mean_val != 0 else np.nan,
        }

    return length, n_bins, results


def compute_surfaces(data: np.ndarray, n_jobs=-1) -> dict:
    """Compute TE surfaces for different measures."""
    surfaces = {
        name: {
            NORM_STD: np.zeros((len(length_range), len(bin_range))),
            MEAN: np.zeros((len(length_range), len(bin_range))),
        }
        for name in ["TE", "NTE", "logNTE"]
    }

    params = list(product(range(len(length_range)), range(len(bin_range))))

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_te_for)(data, length_range[i], bin_range[j]) for i, j in params
    )

    for length, n_bins, measures in results:
        i = np.where(length_range == length)[0][0]
        j = np.where(bin_range == n_bins)[0][0]

        for measure_name, measure_vals in measures.items():
            surfaces[measure_name][MEAN][i, j] = measure_vals[MEAN]
            surfaces[measure_name][NORM_STD][i, j] = measure_vals[NORM_STD]

    return surfaces


def save_surfaces(surfaces: dict, filename: str):
    """Save computed surfaces to a pickle file."""
    save_path = SURFACE_DATA_DIR / filename
    with open(save_path, "wb") as f:
        pickle.dump(surfaces, f)


def load_surfaces(filename: str) -> dict:
    """Load computed surfaces from a pickle file."""
    load_path = SURFACE_DATA_DIR / filename
    with open(load_path, "rb") as f:
        return pickle.load(f)


def plot_measure_surface(
    measure_vals: np.ndarray,
    bins: np.ndarray,
    lengths: np.ndarray,
    measure_name: str,
    filename: str,
    **plot_kwargs,
):
    """Plot measure surface using trisurf with customizable plot parameters."""
    x_vals, y_vals = np.meshgrid(bins, lengths)
    fig = plt.figure(figsize=plot_kwargs.get("figsize", (12, 8)))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    plot_params = {
        "vmin": 0,
        "vmax": np.nanmax(measure_vals) * 1.05,
        "cmap": "viridis",
        "alpha": 0.75,
        "linewidth": 0.5,
        "antialiased": True,
        "shade": True,
    }
    plot_params.update(plot_kwargs.get("surface_params", {}))

    surface = ax.plot_surface(x_vals, y_vals, measure_vals, **plot_params)

    ax.set_xlabel(plot_kwargs.get("xlabel", "Number of bins"))
    ax.set_ylabel(plot_kwargs.get("ylabel", "Sample length"))
    ax.set_zlabel(plot_kwargs.get("zlabel", measure_name), rotation=90)

    ax.view_init(elev=45, azim=-80)

    if plot_kwargs.get("add_colorbar", False):
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    plt.savefig(
        SURFACE_PLOT_DIR / filename,
        dpi=plot_kwargs.get("dpi", 300),
        bbox_inches="tight",
    )
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
        filename = f"{name.lower()}x{N_MAPS}_surface_{N_LENS}data_{N_BINS}bins.png"
        plot_measure_surface(
            surface[MEAN],
            bin_range,
            length_range,
            name,
            f"{MEAN}_{filename}",
            **plot_kwargs,
        )
        plot_measure_surface(
            surface[NORM_STD],
            bin_range,
            length_range,
            name,
            f"{NORM_STD}_{filename}",
            **plot_kwargs,
        )


def main():
    """Scan (N)TE for samples and bin."""
    for map_name, _ in maps.items():
        print("Evaluating map", map_name)
        filename = (
            f"surfaces_{map_name}_{EPS}eps_x{N_MAPS}"
            f"_surface_{N_LENS}data_{N_BINS}bins.pkl"
        )
        # Check if saved surfaces exist
        if not (SURFACE_DATA_DIR / filename).exists():
            print("Computing and saving surfaces...")
            surfaces = compute_and_save_surfaces(map_name, filename)
        else:
            print("Loading pre-computed surfaces...")
            surfaces = load_surfaces(filename)

        # Plot surfaces with default parameters
        plot_all_surfaces(surfaces, {})


if __name__ == "__main__":
    main()
