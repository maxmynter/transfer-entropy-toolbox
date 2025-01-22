"""Analyze transfer entropy using different binning methods."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from binning_constants import (
    BINNING_METHODS,
    CRITERION_BINS_PATTERN,
    DATA_DIR,
    DEFAULT_MAP,
    EPS,
    LAG,
    N_MAPS,
    N_TRANSIENT,
    PLOT_DIR,
    PLOT_STYLES,
    RESULTS_FILE_PATTERN,
    SAMPLE_SIZES,
    SEED,
    STD_PLOT_PATTERN,
    VIOLIN_PLOT_PATTERN,
)

from te_toolbox.entropies import (
    logn_normalized_transfer_entropy,
    normalized_transfer_entropy,
    transfer_entropy,
)
from te_toolbox.systems.lattice import CMLConfig, CoupledMapLatticeGenerator

# Configure plotting
plt.rcParams.update(PLOT_STYLES)
sns.set_style("whitegrid")


def generate_cml_data(sample_size):
    """Generate coupled map lattice data."""
    config = CMLConfig(
        map_function=DEFAULT_MAP,
        n_maps=N_MAPS,
        coupling_strength=EPS,
        n_steps=sample_size,
        warmup_steps=N_TRANSIENT,
        seed=SEED,
    )
    return CoupledMapLatticeGenerator(config).generate().lattice


def compute_te_for_method(data, method_func):
    """Compute TE values using specified binning method."""
    te_vals = []
    nte_vals = []
    lognte_vals = []

    for k in range(N_MAPS - 1):
        pair_data = data[:, [k, k + 1]]

        try:
            bins = method_func(pair_data.flatten())

            te = transfer_entropy(pair_data, bins, LAG, at=(1, 0))
            nte = normalized_transfer_entropy(pair_data, bins, LAG, at=(1, 0))
            lognte = logn_normalized_transfer_entropy(pair_data, bins, LAG, at=(1, 0))

            te_vals.append(te)
            nte_vals.append(nte)
            lognte_vals.append(lognte)
        except Exception as e:
            print(f"Error with method {method_func.__name__}: {e!s}")
            te_vals.append(np.nan)
            nte_vals.append(np.nan)
            lognte_vals.append(np.nan)

    return np.array(te_vals), np.array(nte_vals), np.array(lognte_vals)


def analyze_binning_methods(sample_size, n_jobs=-1):
    """Analyze all binning methods for a given sample size."""
    results_file = DATA_DIR / RESULTS_FILE_PATTERN.format(sample_size)

    # Check if results already exist
    if results_file.exists():
        with open(results_file, "rb") as f:
            return pickle.load(f)

    data = generate_cml_data(sample_size)

    results = {}
    for method_name, method_func in BINNING_METHODS.items():
        te_vals, nte_vals, lognte_vals = compute_te_for_method(data, method_func)
        results[method_name] = {
            "TE": te_vals[~np.isnan(te_vals)],
            "NTE": nte_vals[~np.isnan(nte_vals)],
            "logNTE": lognte_vals[~np.isnan(lognte_vals)],
        }

    # Save results
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results


def plot_violin_comparison(results_by_size, metric="TE"):
    """Create violin plots comparing binning methods."""
    plt.figure(figsize=PLOT_STYLES["figure.figsize"])

    # Prepare data for plotting
    plot_data = []
    for size, results in results_by_size.items():
        for method, vals in results.items():
            method_vals = vals[metric]
            plot_data.extend([(method, val, size) for val in method_vals])

    df = pd.DataFrame(plot_data, columns=["Method", metric, "Sample Size"])

    # Create violin plot
    sns.violinplot(data=df, x="Method", y=metric, hue="Sample Size")
    plt.xticks(rotation=45)
    plt.title(f"{metric} Distribution by Binning Method")
    plt.tight_layout()

    # Save plot
    plot_file = PLOT_DIR / VIOLIN_PLOT_PATTERN.format(metric.lower(), max(SAMPLE_SIZES))
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_std_comparison(results_by_size, metric="TE"):
    """Plot standard deviation of TE/NTE values for each method."""
    plt.figure(figsize=(12, 6))

    for method in BINNING_METHODS:
        stds = [np.std(results[method][metric]) for results in results_by_size.values()]
        plt.plot(SAMPLE_SIZES, stds, "o-", label=method)

    plt.xscale("log")
    plt.xlabel("Sample Size")
    plt.ylabel(f"Standard Deviation of {metric}")
    plt.title(f"{metric} Variability by Binning Method")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_file = PLOT_DIR / STD_PLOT_PATTERN.format(metric.lower())
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_criterion_comparison(results_by_size, metric="TE"):
    """Plot mean values with error bars against sample size for a single metric."""
    plt.figure(figsize=(12, 6))

    sizes = list(results_by_size.keys())
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(BINNING_METHODS))
    )  # Color for each method

    for idx, method in enumerate(BINNING_METHODS.keys()):
        means = []
        stds = []
        for size in sizes:
            values = results_by_size[size][method][metric]
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.errorbar(
            sizes,
            means,
            yerr=stds,
            fmt="o-",
            color=colors[idx],
            label=method,
            capsize=5,
            alpha=0.7,
        )

    plt.xscale("log")
    plt.xticks(sizes, sizes)

    plt.xlabel("Sample Size")
    plt.ylabel(f"{metric} Value")
    plt.title(f"{metric} by Sample Size and Method")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_file = PLOT_DIR / CRITERION_BINS_PATTERN.format(metric.lower())
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run binning analysis."""
    results_by_size = {}
    for size in SAMPLE_SIZES:
        print(f"Analyzing sample size: {size}")
        results_by_size[size] = analyze_binning_methods(size)

    for metric in ["TE", "NTE", "logNTE"]:
        # plot_violin_comparison(results_by_size, metric)
        # plot_std_comparison(results_by_size, metric)
        plot_criterion_comparison(results_by_size, metric)


if __name__ == "__main__":
    main()
