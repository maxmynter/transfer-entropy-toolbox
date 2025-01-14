"""Plot the Epsilon scan results."""

import gc
import os
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import psutil
import seaborn as sns
from eps_scan_constants import (
    EPS_DATA_DIR,
    EPSILONS,
    LAG,
    N_BINS,
    N_ITER,
    N_MAPS,
    PLOT_PATH,
    RELATIVE_NOISE_AMPLITUDE,
    SEED,
)

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)
from te_toolbox.preprocessing import noisify, remap_to
from te_toolbox.systems.lattice import CoupledMapLattice
from te_toolbox.systems.maps import BellowsMap, ExponentialMap, LogisticMap, TentMap

# Configure plotting
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stix",
    }
)
sns.set()


def log_memory_usage(location: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    print(
        f"Memory usage at {location}: {process.memory_info().rss / 1024 / 1024:.2f} MB"
    )


def plot_results(results, plot_prefix, n_maps, n_iter, epsilons, n_bins):  # noqa: PLR0913, Need these args here
    """Plot the results with error bars."""
    # Transfer Entropy plot
    res_te_mean = np.mean(results["te"], axis=0)
    max_res_te_bin_idx = np.argmax(res_te_mean)
    plot_measure(
        epsilons,
        results["te"],
        results["te_err"],
        n_bins,
        max_res_te_bin_idx,
        "TE",
        plot_prefix,
        n_maps,
        n_iter,
    )

    # H-normalized TE plot
    res_hnte_mean = np.mean(results["hnte"], axis=0)
    max_res_hnte_mean_bin_idx = np.argmax(res_hnte_mean)
    plot_measure(
        epsilons,
        results["hnte"],
        results["hnte_err"],
        n_bins,
        max_res_hnte_mean_bin_idx,
        "HNTE",
        plot_prefix,
        n_maps,
        n_iter,
    )

    # Log-normalized TE plot
    res_lognte_mean = np.mean(results["lognte"], axis=0)
    max_res_lognte_mean_bin_idx = np.argmax(res_lognte_mean)
    plot_measure(
        epsilons,
        results["lognte"],
        results["lognte_err"],
        n_bins,
        max_res_lognte_mean_bin_idx,
        "logNTE",
        plot_prefix,
        n_maps,
        n_iter,
    )


def plot_measure(  # noqa: PLR0913, Need these args here
    epsilons, values, errors, n_bins, max_idx, measure_name, plot_prefix, n_maps, n_iter
):
    """Plot a single measure with error bars."""
    plt.figure()
    for b in range(len(n_bins)):
        lin = "dashed" if b > max_idx else "solid"
        plt.errorbar(
            epsilons,
            values[:, b],
            yerr=errors[:, b],
            linestyle=lin,
            label=f"{n_bins[b]} Bins",
        )
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(measure_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}SchreiberReprod_{measure_name}_{n_maps}_maps_{n_iter}_steps.png",
        dpi=300,
    )
    plt.close()


def pairwise_tes(
    pair: npt.NDArray, bins: npt.NDArray
) -> tuple[np.float64, np.float64, np.float64]:
    """Process all TE measures for 2 time series."""
    te = transfer_entropy(pair, bins, LAG, at=(1, 0))
    nte = normalized_transfer_entropy(pair, bins, LAG, at=(1, 0))
    lognte = logn_normalized_transfer_entropy(pair, bins, LAG, at=(1, 0))
    return te, nte, lognte


def analyze_map(  # Script analysis takes some lines sometimes....
    map_name,
    plot_prefix="",
    data_dir=EPS_DATA_DIR,
    add_noise_flag=True,
    gaussian_remap_flag=True,
):
    """Run and plot analysis of chaotic map."""
    tracemalloc.start()
    log_memory_usage("Start of analyze_map")
    # Initialize results arrays
    results = {
        "te": np.zeros((len(EPSILONS), len(N_BINS))),
        "hnte": np.zeros((len(EPSILONS), len(N_BINS))),
        "lognte": np.zeros((len(EPSILONS), len(N_BINS))),
        "te_err": np.zeros((len(EPSILONS), len(N_BINS))),
        "hnte_err": np.zeros((len(EPSILONS), len(N_BINS))),
        "lognte_err": np.zeros((len(EPSILONS), len(N_BINS))),
    }

    rng = np.random.default_rng(SEED)

    for e, eps in enumerate(EPSILONS):
        print(f"Processing epsilon {eps}")
        log_memory_usage(f"Start of epsilon {eps}")
        # Load and process data
        data = CoupledMapLattice.load(
            data_dir
            / CoupledMapLattice.generate_default_filename(
                map_name, N_MAPS, N_ITER, eps, SEED
            )
        )
        log_memory_usage("After data load")
        lattice = data.lattice
        log_memory_usage("After getting lattice")

        if add_noise_flag:
            lattice = noisify(
                lattice,
                noise_distribution="normal",
                amplitude=RELATIVE_NOISE_AMPLITUDE,
                rng=rng,
            )
            log_memory_usage("After adding noise")

        if gaussian_remap_flag:
            gaussian_samples = rng.normal(size=lattice.shape)
            lattice = remap_to(lattice, gaussian_samples, rng)
            log_memory_usage("After gaussian remap")
            del gaussian_samples
            gc.collect()

        for b, n_bins in enumerate(N_BINS):
            log_memory_usage(f"Start of bins {n_bins}")
            bins = np.linspace(np.min(lattice), np.max(lattice), n_bins + 1)

            te_vals = []
            nte_vals = []
            lognte_vals = []

            for k in range(N_MAPS - 1):  # -1 because we look at pairs
                pair_data = lattice[:, [k, k + 1]]

                te, nte, lognte = pairwise_tes(pair_data, bins)
                te_vals.append(te)
                nte_vals.append(nte)
                lognte_vals.append(lognte)

            results["te"][e, b] = np.mean(te_vals)
            results["hnte"][e, b] = np.mean(nte_vals)
            results["lognte"][e, b] = np.mean(lognte_vals)
            results["te_err"][e, b] = np.std(te_vals)
            results["hnte_err"][e, b] = np.std(nte_vals)
            results["lognte_err"][e, b] = np.std(lognte_vals)

            log_memory_usage(f"End of bins {n_bins}")

        del lattice
        del data

        gc.collect()
        log_memory_usage(f"End of epsilon {eps}")

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[ Top 10 memory users ]")
        for stat in top_stats[:10]:
            print(stat)

    tracemalloc.stop()
    return results


def main():
    """Plot the Epsilon Scan results."""
    maps = {
        "LogisticMap(r=4)": LogisticMap(r=4),
        "BellowsMap(r=5,b=6)": BellowsMap(r=5, b=6),
        "ExponentialMap(r=4)": ExponentialMap(r=4),
        "TentMap(r=2)": TentMap(r=2),
    }

    for map_name in maps:
        print(f"__________________{map_name}")
        plot_prefix = (
            str(PLOT_PATH)
            + "/"
            + (
                f"Noise_{RELATIVE_NOISE_AMPLITUDE}_GaussianRemap_"
                f"{N_ITER}Data_{len(EPSILONS)}_eps_{map_name}_"
            )
        )
        results = analyze_map(map_name, plot_prefix)

        plot_results(results, plot_prefix, N_MAPS, N_ITER, EPSILONS, N_BINS)


if __name__ == "__main__":
    main()
