"""Plot the TE pairs for the fixed bins."""

import re

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from futures_constants import DATA_PATH, PLOT_PATH
from wrangling.columns import Columns, TEColumns

sns.set()
SUBTRACT_SURRO = False

asym_filenames = [
    (
        "tents_ts_1Lag_fn=fixed_2_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_4_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_8_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_16_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_32_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_64_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_128_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_256_bins_over_range_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
]

sym_filenames = [
    (
        "tents_ts_1Lag_fn=fixed_2_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_4_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_8_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_16_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_32_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_64_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_128_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
    (
        "tents_ts_1Lag_fn=fixed_256_bins_sym_abt_0_5window_1stepsourcecol=log "
        "returns (5m).csv"
    ),
]


def extract_bin_size(filename: str) -> int:
    """Extract number of bins from filename."""
    match = re.search(r"fixed_(\d+)_bins", filename)
    if match:
        return int(match.group(1))
    raise Exception("Could not parse number of bins from filename")


def read_dataframes(filenames: list[str]) -> dict[int, pl.DataFrame]:
    """Read dataframes and store them by bin size."""
    dataframes = {}
    for filename in filenames:
        bin_size = extract_bin_size(filename)
        if bin_size > 0:
            print(f"Reading file with {bin_size} bins: {filename}")
            try:
                df = pl.read_csv(DATA_PATH / filename)
                dataframes[bin_size] = df
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return dataframes


def plot_specific_pairs_by_bins(
    dataframes: dict[int, pl.DataFrame],
    pairs: list[tuple],
    bin_strategy: str,
    use_surrogate: bool = True,
):
    """
    Plot specific TE pairs for different bin sizes.

    Args:
        dataframes: Dictionary mapping bin sizes to dataframes
        pairs: List of (source, target) pairs to plot
        bin_strategy: String describing the binning strategy behaviour
                      about zero
        use_surrogate: Whether to subtract surrogate TE (default is True)

    """
    # Sort bin sizes
    bin_sizes = sorted(dataframes.keys())

    # Create subplots for each pair
    for src, tgt in pairs:
        fig, axes = plt.subplots(
            len(bin_sizes), 1, figsize=(12, 3 * len(bin_sizes)), sharex=True
        )
        fig.suptitle(
            f"Transfer Entropy {src.base} → {tgt.base} with {bin_strategy} binning",
            fontsize=16,
        )

        # Get column names for raw TE and surrogate (bs)
        raw_col = TEColumns.get_te_column_name(src, tgt)
        bs_col = TEColumns.get_bs_column_name(src, tgt) if use_surrogate else None

        # Plot each bin size
        for i, bin_size in enumerate(bin_sizes):
            ax = axes[i] if len(bin_sizes) > 1 else axes
            df = dataframes[bin_size]

            # Calculate values to plot (raw or raw-bs)
            if use_surrogate and bs_col in df.columns:
                values = (df[raw_col] - df[bs_col]).map_elements(
                    lambda x: max(x, 0), return_dtype=pl.Float64
                )
                y_label = f"TE - BS Surrogate ({bin_size} bins)"
            else:
                values = df[raw_col]
                y_label = f"TE ({bin_size} bins)"

            # Plot the values
            ax.plot(df[Columns.Date], values, label=f"{bin_size} bins", linewidth=1.5)

            # Add horizontal line at y=0 for reference
            ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)

            # Customize plot
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

            # Add bin size text to the right
            ax.text(
                0.99,
                0.01,
                f"{bin_size} bins",
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="bottom",
            )

        # Customize the bottom axis
        bottom_ax = axes[-1] if len(bin_sizes) > 1 else axes
        bottom_ax.set_xlabel("Date")

        # Format x-axis
        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        # Save the figure
        plt.savefig(
            PLOT_PATH
            / f"te_{src.base}_to_{tgt.base}_{bin_strategy}_{len(bin_sizes)}_bins.png"
        )
        plt.close()


def create_combined_plot(
    dataframes: dict[int, pl.DataFrame],
    pairs: list[tuple],
    bin_strategy: str,
    selected_bins: list[int],
    use_surrogate: bool = True,
):
    """
    Create a single plot showing multiple bin sizes for comparison.

    Args:
        dataframes: Dictionary mapping bin sizes to dataframes
        pairs: List of (source, target) pairs to plot
        bin_strategy: String describing the binning strategy
        selected_bins: List of bin sizes to include in the plot
        use_surrogate: Whether to subtract surrogate TE

    """
    # Filter to only selected bin sizes
    selected_dfs = {k: v for k, v in dataframes.items() if k in selected_bins}

    # Create a plot for each pair
    for src, tgt in pairs:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get column names
        raw_col = TEColumns.get_te_column_name(src, tgt)
        bs_col = TEColumns.get_bs_column_name(src, tgt) if use_surrogate else None

        # Plot each selected bin size
        for bin_size, df in sorted(selected_dfs.items()):
            # Calculate values
            if use_surrogate and bs_col in df.columns:
                values = (df[raw_col] - df[bs_col]).map_elements(
                    lambda x: max(0, x), return_dtype=pl.Float64
                )
            else:
                values = df[raw_col]

            # Plot with distinct line style for visibility
            ax.plot(df[Columns.Date], values, label=f"{bin_size} bins", linewidth=1.5)

        # Add horizontal line at y=0
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.3)

        ax.set_xlabel("Date")

        if use_surrogate:
            ax.set_ylabel("TE - BS Surrogate")
        else:
            ax.set_ylabel("Transfer Entropy")

        # Add legend and grid
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        bin_str = "_".join(str(b) for b in selected_bins)
        plt.savefig(
            PLOT_PATH
            / f"combined_te_{src.base}_to_{tgt.base}_{bin_strategy}_{bin_str}.png"
        )
        plt.close()


if __name__ == "__main__":
    # Define specific pairs to analyze
    specific_pairs = [
        (Columns.VG, Columns.HS),  # VG1 -> HI1
        (Columns.VG, Columns.ES),  # VG1 -> ES1
        (Columns.HS, Columns.VG),  # HI1 -> VG1 (bidirectional check)
        (Columns.ES, Columns.VG),  # ES1 -> VG1 (bidirectional check)
    ]

    # Read in all the dataframes
    print("Reading asymmetric binning files...")
    asym_dfs = read_dataframes(asym_filenames)

    print("Reading symmetric binning files...")
    sym_dfs = read_dataframes(sym_filenames)

    # Plot individual bin sizes
    print("Creating plots for asymmetric binning...")
    plot_specific_pairs_by_bins(asym_dfs, specific_pairs, "equisize", SUBTRACT_SURRO)

    print("Creating plots for symmetric binning...")
    plot_specific_pairs_by_bins(sym_dfs, specific_pairs, "on_zero", SUBTRACT_SURRO)

    # Create combined plots with selected bin sizes for comparison
    selected_bins = [
        2,
        4,
        8,
        16,
        64,
        128,
        256,
    ]  # Selected bin sizes to highlight differences

    print("Creating combined plots for asymmetric binning...")
    create_combined_plot(
        asym_dfs, specific_pairs, "equisize", selected_bins, SUBTRACT_SURRO
    )

    print("Creating combined plots for symmetric binning...")
    create_combined_plot(
        sym_dfs, specific_pairs, "on_zero", selected_bins, SUBTRACT_SURRO
    )

    print("All plots created successfully!")
