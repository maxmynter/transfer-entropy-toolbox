"""Analyse the Transfer entropy data."""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from futures_constants import DATA_PATH, PLOT_PATH, TentCalcConfig
from futures_main import filename
from wrangling import Cols
from wrangling.columns import TEColumns

config = TentCalcConfig()


class ColPrefix(Enum):
    """Subtract column with prefix from tent - None if empty."""

    FT = "ft_surr_"
    BS = "bs_surr_"
    RAW = ""


def create_te_plot(
    df: pl.DataFrame, reduction: ColPrefix, config: TentCalcConfig
) -> None:
    """
    Create a transfer entropy plot for the given reduction type.

    Args:
        df: Input DataFrame containing TE data
        reduction: Type of reduction to apply (FT, BS, or RAW)
        config: Configuration object containing calculation parameters

    """
    cols = Cols.get_all_instruments()
    pairs = [[(src, tgt) for src in cols if src != tgt] for tgt in cols]

    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for idx, src_tgt in enumerate(pairs):
        ax = axs[idx]
        for src, tgt in src_tgt:
            column = TEColumns.get_te_column_name(src, tgt)
            values = (
                np.maximum(0, df[column] - df[f"{reduction.value}{column}"])
                if reduction != ColPrefix.RAW
                else df[column]
            )
            ax.plot(df[Cols.Date], values, label=f"x = {src.base}")

        ax.set_ylabel(f"x → {tgt.base}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, loc="upper right", fontsize="small")

    axs[-1].set_xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot with reduction type in filename
    plt.savefig(PLOT_PATH / f"{reduction.name}_{filename}_pairwise_tents.png")
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Set style once for all plots
    sns.set()

    # Read and prepare data
    csv_filename = filename + ".csv"
    print(f"Reading in {csv_filename}")
    df = pl.read_csv(DATA_PATH / csv_filename)
    df = df[config.WINDOW_SIZE :]
    print(df.head())

    # Create plots for each reduction type
    config = TentCalcConfig()
    for reduction in ColPrefix:
        print(f"Creating plot for {reduction.name} reduction")
        create_te_plot(df, reduction, config)
