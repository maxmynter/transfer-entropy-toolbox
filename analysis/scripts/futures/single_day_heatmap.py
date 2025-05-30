"""Create heatmap for TE between instruments on specific dates."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from futures_constants import DATA_PATH, PLOT_PATH, TentCalcConfig
from wrangling.columns import Cols

sns.set()

# Use the same CSV as your network code
csv_filename = (
    "tents_ts_1Lag_fn=get_maximised_te_fixed_range_"
    f"{TentCalcConfig.WINDOW_SIZE}window_1stepsourcecol=log returns (5m).csv"
)

df = pd.read_csv(DATA_PATH / csv_filename)
df["Date"] = pd.to_datetime(df["Date"])

cols = Cols.get_all_instruments()
assets = [col.base for col in cols]


def create_te_matrix(date_str, data):
    """Create the matrix to plot the heat map."""
    date_data = data[data["Date"] == date_str]
    if len(date_data) == 0:
        print(f"No data found for {date_str}")
        return None

    matrix = np.zeros((len(assets), len(assets)))

    for i, source in enumerate(assets):
        for j, target in enumerate(assets):
            if source != target:
                col_name = f"{source}->{target}"
                if col_name in date_data.columns:
                    matrix[i, j] = date_data[col_name].iloc[0]

    return matrix


# Rest of your heatmap code stays the same...
jan_30_matrix = create_te_matrix("2020-01-30", df)
mar_16_matrix = create_te_matrix("2020-03-16", df)

if jan_30_matrix is not None and mar_16_matrix is not None:
    mask = np.eye(len(assets), dtype=bool)

    # Find the overall min and max for consistent color scaling
    # Only consider non-diagonal elements
    jan_values = jan_30_matrix[~mask]
    mar_values = mar_16_matrix[~mask]
    vmin = min(jan_values.min(), mar_values.min())
    vmax = max(jan_values.max(), mar_values.max())

    # January 30 heatmap
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(
        jan_30_matrix,
        xticklabels=assets,
        yticklabels=assets,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        ax=ax1,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Transfer Entropy"},
    )
    ax1.set_xlabel("Target", fontweight="bold")
    ax1.set_ylabel("Source", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        PLOT_PATH / "transfer_entropy_heatmap_2020-01-30.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # March 16 heatmap
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(
        mar_16_matrix,
        xticklabels=assets,
        yticklabels=assets,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        mask=mask,
        ax=ax2,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Transfer Entropy"},
    )
    ax2.set_xlabel("Target", fontweight="bold")
    ax2.set_ylabel("Source", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        PLOT_PATH / "transfer_entropy_heatmap_2020-03-16.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Heatmaps saved:")
    print("- transfer_entropy_heatmap_2020-01-30.png")
    print("- transfer_entropy_heatmap_2020-03-16.png")
    print(f"Color scale range: {vmin:.3f} to {vmax:.3f}")
