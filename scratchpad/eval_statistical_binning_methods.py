"""Cross check statistical binning criterion cost functions."""

import matplotlib.pyplot as plt
import numpy as np

from te_toolbox.binning.statistical import (
    aic_cost,
    aicc_cost,
    bic_cost,
    knuth_cost,
    shimazaki_cost,
)


def plot_cost_functions(data: np.ndarray, min_bins: int = 3):
    """Plot cost function values against number of bins for different criteria."""
    max_bins = len(data)

    # Define cost functions and their properties
    cost_functions = {
        "Knuth": (knuth_cost, False),  # (function, should_minimize)
        "Shimazaki": (shimazaki_cost, True),
        "AIC": (aic_cost, True),
        "BIC": (bic_cost, True),
        "AICc": (aicc_cost, True),
    }

    n_bins_range = range(min_bins, max_bins)

    for name, (cost_func, minimize) in cost_functions.items():
        plt.figure(figsize=(15, 10))
        costs: list[float] = []
        for n_bins in n_bins_range:
            bins = np.linspace(np.min(data), np.max(data), n_bins)
            hist, _ = np.histogram(data, bins)
            cost = cost_func(hist, bins)
            # Normalize cost to make them comparable in plot
            if len(costs) > 0:
                cost = cost / abs(costs[0])
            costs.append(cost)

        plt.plot(
            n_bins_range, costs, label=f"{name} {'(min)' if minimize else '(max)'}"
        )
        # plt.yscale("log")

        plt.xlabel("Number of Bins")
        plt.ylabel("Normalized Cost Value")
        plt.title(f"{name} vs Number of Bins for Normal Distribution (n = {len(data)})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.legend()

        plt.savefig(f"{name}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Run Script."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.normal(0, 1, n_samples)

    plot_cost_functions(data, min_bins=3)


if __name__ == "__main__":
    main()
