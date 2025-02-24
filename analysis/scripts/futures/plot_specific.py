"""Plot specific TE pairs with interesting features."""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from futures_constants import (
    DATA_PATH,
    PLOT_PATH,
    TE,
    TentCalcConfig,
    generate_filename,
)
from futures_main import get_bootstrap_maximised_te
from wrangling.columns import Columns, ReturnType, TEColumns

sns.set()


class SpecificConfig(TentCalcConfig):
    """Specific Config Values for detailed analysis."""

    TE = TE.LOGN
    LAG = 1
    WINDOW_SIZE = 7
    WINDOW_STEP = 1
    on_column = ReturnType.LOG


filename = generate_filename(SpecificConfig(), get_bootstrap_maximised_te)

onto_co = [
    TEColumns.get_bs_column_name(Columns.ES, Columns.CO),
    TEColumns.get_bs_column_name(Columns.NK, Columns.CO),
    TEColumns.get_bs_column_name(Columns.VG, Columns.CO),
    TEColumns.get_bs_column_name(Columns.HS, Columns.CO),
]

vg_hi = [
    TEColumns.get_bs_column_name(Columns.VG, Columns.HS),
    TEColumns.get_bs_column_name(Columns.HS, Columns.VG),
]


def plot_specific_te_pairs(
    df: pl.DataFrame, onto_co: list[str], vg_hi: list[str], filename="specific_te.png"
):
    """Plot specific TE pairs."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for col in onto_co:
        ax1.plot(df[Columns.Date], df[col], label=col.split("_")[2][:3], alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for col in vg_hi:
        ax2.plot(
            df[Columns.Date],
            df[col],
            label=f"{col.split('_')[2][:3]} → {col.split('_')[2][-3:]}",
            alpha=0.7,
        )
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xticks(rotation=45)

    ax2.set_title("Bidirectional Information Flow between EuroStoxx and Hang Seng")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Date")
    ax1.set_ylabel(
        "(NTE x → CO1) - Bootstrapped sample NTE",
    )
    ax2.set_title("Bidirectional Information Flow between EuroStoxx and Hang Seng")
    ax2.legend()
    ax2.grid(True)

    ax2.set_ylabel("NTE - Bootstrapped sample NTE")

    plt.tight_layout()
    plt.savefig(PLOT_PATH / filename)
    plt.close()


if __name__ == "__main__":
    df = pl.read_csv(DATA_PATH / (filename + ".csv"))
    cols = []
    cols.extend(onto_co)
    cols.extend(vg_hi)
    df = df.select([Columns.Date, *cols])

    plot_specific_te_pairs(df, onto_co, vg_hi, "specific_te_covid.png")
