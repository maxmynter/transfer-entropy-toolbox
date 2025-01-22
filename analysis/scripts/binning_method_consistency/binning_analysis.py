"""Analyze transfer entropy using different binning methods."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from binning_constants import (
    BINNING_METHODS,
    CRITERION_BINS_PATTERN,
    DATA_DIR,
    DEFAULT_MAP,
    EPS,
    LAG,
    METRICS,
    N_MAPS,
    N_TRANSIENT,
    PLOT_DIR,
    PLOT_STYLES,
    RESULTS_FILE_PATTERN,
    SAMPLE_SIZES,
    SEED,
)
from joblib import Parallel, delayed
from metric_enum import Metric

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
    metric_vals = {metric: [] for metric in METRICS}

    for k in range(N_MAPS - 1):
        pair_data = data[:, [k, k + 1]]

        try:
            bins = method_func(pair_data.flatten())

            for metric in METRICS:
                metric_vals[metric].append(
                    metric.compute(pair_data, bins, LAG, at=(1, 0))
                )
        except Exception as e:
            print(f"Error with method {method_func.__name__}: {e!s}")
            for metric in METRICS:
                metric_vals[metric].append(np.nan)

    return {metric: np.array(vals) for metric, vals in metric_vals.items()}


def analyze_binning_methods(sample_size, n_jobs=-1):
    """Analyze all binning methods for a given sample size."""
    filename = (
        "_".join(str(m) for m in METRICS)
        + "_"
        + RESULTS_FILE_PATTERN.format(sample_size)
    )
    results_file = DATA_DIR / filename

    # Check if results already exist
    if results_file.exists():
        with open(results_file, "rb") as f:
            return pickle.load(f)

    data = generate_cml_data(sample_size)

    def process_method(method_item):
        method_name, method_func = method_item
        metric_vals = compute_te_for_method(data, method_func)
        return method_name, {
            metric: vals[~np.isnan(vals)] for metric, vals in metric_vals.items()
        }

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_method)(item) for item in BINNING_METHODS.items()
    )

    results = dict(results_list)

    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results


def plot_criterion_comparison_shaded_subplots_by_type(results_by_size, metric: Metric):
    """Plot methods side by side: statistical vs rule-based."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    sizes = list(results_by_size.keys())

    rule_methods = ["Doane", "Freedman-Diaconis", "Rice", "Scott", "Sqrt-n", "Sturges"]
    statistical_methods = [
        "AIC",
        "BIC",
        "Knuth",
        "Shimazaki",
    ]

    colors1 = plt.cm.tab10(np.linspace(0, 1, len(rule_methods)))
    colors2 = plt.cm.tab10(np.linspace(0, 1, len(statistical_methods)))

    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "+", "x"]

    # Plot rule-based methods
    for idx, method in enumerate(rule_methods):
        means_list = []
        stds_list = []
        for size in sizes:
            values = results_by_size[size][method][metric]
            means_list.append(np.mean(values))
            stds_list.append(np.std(values))
        means = np.array(means_list)
        stds = np.array(stds_list)

        ax1.plot(sizes, means, f"{markers[idx]}-", color=colors1[idx], label=method)
        ax1.fill_between(
            sizes, means - stds, means + stds, color=colors1[idx], alpha=0.2
        )

    # Plot statistical methods
    for idx, method in enumerate(statistical_methods):
        means_list = []
        stds_list = []
        for size in sizes:
            values = results_by_size[size][method][metric]
            means_list.append(np.mean(values))
            stds_list.append(np.std(values))
        means = np.array(means_list)
        stds = np.array(stds_list)

        ax2.plot(sizes, means, f"{markers[idx]}-", color=colors2[idx], label=method)
        ax2.fill_between(
            sizes, means - stds, means + stds, color=colors2[idx], alpha=0.2
        )

    max_ent_label = f"Max {metric}"
    ax2.errorbar(
        sizes,
        [np.mean(results_by_size[size][max_ent_label][metric]) for size in sizes],
        yerr=[np.std(results_by_size[size][max_ent_label][metric]) for size in sizes],
        fmt="o-",
        color="red",
        label=max_ent_label,
        capsize=5,
        linewidth=2,
        alpha=0.8,
        markersize=8,
    )
    for ax in [ax1, ax2]:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"{metric} Value")
        ax.legend(loc="lower right")

    ax2.set_xlabel("Sample Size")

    ax1.set_title("Rule-based Methods")
    ax2.set_title("Statistical Methods")

    plt.suptitle(f"{metric} by Sample Size and Method Type")
    plt.tight_layout()

    plot_file = PLOT_DIR / CRITERION_BINS_PATTERN.format(metric.lower())
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Run binning analysis."""
    results_by_size = {}
    for size in SAMPLE_SIZES:
        print(f"Analyzing sample size: {size}")
        results_by_size[size] = analyze_binning_methods(size)

    for metric in METRICS:
        plot_criterion_comparison_shaded_subplots_by_type(results_by_size, metric)


if __name__ == "__main__":
    main()
