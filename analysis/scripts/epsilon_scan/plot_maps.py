"""Plot the Epsilon scan results."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from eps_scan_constants import (
    EPSILONS,
    LAG,
    N_BINS,
    N_ITER,
    N_MAPS,
    OUTPUT_DIR,
    RELATIVE_NOISE_AMPLITUDE,
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


def analyze_map(  # noqa: PLR0915 # Script analysis takes some lines sometime....
    map_name,
    plot_prefix="",
    data_dir=OUTPUT_DIR,
    add_noise_flag=True,
    gaussian_remap_flag=True,
):
    """Run and plot analysis of chaotic map."""
    # Initialize results arrays
    shape = (len(EPSILONS), len(N_BINS))
    res_te = np.zeros(shape)
    res_hnte = np.zeros(shape)
    res_lognte = np.zeros(shape)

    res_te_err = np.zeros(shape)
    res_hnte_err = np.zeros(shape)
    res_lognte_err = np.zeros(shape)

    rng = np.random.default_rng(42)

    # For each epsilon
    for e, eps in enumerate(EPSILONS):
        # Load and process data
        data = CoupledMapLattice.load(
            data_dir / f"cml_map={map_name}_{N_MAPS}_x_{N_ITER}eps={eps}_seed=42.npz"
        )
        lattice = data.lattice

        print(f"Processing epsilon {eps}")

        if add_noise_flag:
            lattice = noisify(
                lattice,
                noise_distribution="normal",
                amplitude=RELATIVE_NOISE_AMPLITUDE,
                rng=rng,
            )
            print("Added Noise")

        if gaussian_remap_flag:
            gaussian_samples = rng.normal(size=lattice.shape)
            lattice = remap_to(lattice, gaussian_samples, rng)
            print("Gaussian Remap")

        for b, n_bins in enumerate(N_BINS):
            bins = np.linspace(np.min(lattice), np.max(lattice), n_bins + 1)

            te = []
            nte = []
            lognte = []

            for k in range(N_MAPS - 1):  # -1 because we look at pairs
                pair_data = lattice[:, [k, k + 1]]
                te.append(transfer_entropy(pair_data, bins, LAG)[1, 0])
                nte.append(normalized_transfer_entropy(pair_data, bins, LAG)[1, 0])
                lognte.append(
                    logn_normalized_transfer_entropy(pair_data, bins, LAG)[1, 0]
                )

            res_te[e, b] = np.mean(te)
            res_hnte[e, b] = np.mean(nte)
            res_lognte[e, b] = np.mean(lognte)

            res_te_err[e, b] = np.std(te)
            res_hnte_err[e, b] = np.std(nte)
            res_lognte_err[e, b] = np.std(lognte)

    # Plot results
    res_te_mean = np.mean(res_te, axis=0)
    max_res_te_bin_idx = np.argmax(res_te_mean)

    # Transfer Entropy plot
    for b in range(len(N_BINS)):
        lin = "dashed" if b > max_res_te_bin_idx else "solid"
        plt.errorbar(
            EPSILONS,
            res_te[:, b],
            yerr=res_te_err[:, b],
            linestyle=lin,
            label=f"{N_BINS[b]} Bins",
        )
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("TE")
    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}SchreiberReprod_TE_{N_MAPS}_maps_{N_ITER}_steps.png", dpi=300
    )
    plt.close()

    # H-normalized TE plot
    res_hnte_mean = np.mean(res_hnte, axis=0)
    max_res_hnte_mean_bin_idx = np.argmax(res_hnte_mean)
    for b in range(len(N_BINS)):
        lin = "dashed" if b > max_res_hnte_mean_bin_idx else "solid"
        plt.errorbar(
            EPSILONS,
            res_hnte[:, b],
            yerr=res_hnte_err[:, b],
            linestyle=lin,
            label=f"{N_BINS[b]} Bins",
        )
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("$H$ NTE")
    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}SchreiberReprod_HNTE_{N_MAPS}_maps_{N_ITER}_steps.png", dpi=300
    )
    plt.close()

    # Log-normalized TE plot
    res_lognte_mean = np.mean(res_lognte, axis=0)
    max_res_lognte_mean_bin_idx = np.argmax(res_lognte_mean)
    for b in range(len(N_BINS)):
        lin = "dashed" if b > max_res_lognte_mean_bin_idx else "solid"
        plt.errorbar(
            EPSILONS,
            res_lognte[:, b],
            yerr=res_lognte_err[:, b],
            linestyle=lin,
            label=f"{N_BINS[b]} Bins",
        )
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\log m$ NTE")
    plt.tight_layout()
    plt.savefig(
        f"{plot_prefix}SchreiberReprod_logNTE_{N_MAPS}_maps_{N_ITER}_steps.png", dpi=300
    )
    plt.close()

    return res_te, res_hnte, res_lognte, res_te_err, res_hnte_err, res_lognte_err


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
            f"Noise_{RELATIVE_NOISE_AMPLITUDE}_GaussianRemap_"
            "{N_ITER}Data_{len(EPSILONS)}Eps_{map_name}_"
        )
        analyze_map(map_name, plot_prefix)


if __name__ == "__main__":
    main()
