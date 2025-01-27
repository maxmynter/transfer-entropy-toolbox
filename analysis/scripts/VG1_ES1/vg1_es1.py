"""Analyse the VG1 ES1 Futures data in terms of Transfer Entropy."""

from datetime import datetime

import matplotlib.pyplot as plt
import polars as pl
from constants_vg1_es1 import DATA_PATH, PLOT_PATH


def read_data():
    """Read VG1 ES1 into Polars Data Frame.

    Data are logarithmic returns of two time windows.
    1st window from 08/16/19 to 03/18/20 covering COVID Lockdown onset.
    2nd window from 08/24/20 to 08/03/2021.
    """
    df = pl.read_csv(DATA_PATH / "full_VG1_ES1.csv", try_parse_dates=True)

    df = df.with_columns(pl.col(df.columns[0]).cast(pl.Datetime))

    df = df.select([df.columns[0], "VG1", "ES1"])
    return df


def plot_histogram(df, plot_name="histograms"):
    """Plot histograms of df."""
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.hist(df["VG1"], bins=50, alpha=0.7)
    plt.title("VG1 Distribution")
    plt.subplot(122)
    plt.hist(df["ES1"], bins=50, alpha=0.7)
    plt.title("ES1 Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / f"{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_series(df, plot_name="timeseries"):
    """Plot the full timeseries in df."""
    datetime_col = df.columns[0]
    plt.figure(figsize=(12, 6))
    plt.plot(df[datetime_col], df["VG1"], label="VG1")
    plt.plot(df[datetime_col], df["ES1"], label="ES1")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title("Time Series Evolution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / f"{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def split_on_date(
    df: pl.DataFrame, cutoff: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a df on a datetime in a before and after datetime."""
    datetime_col = df.columns[0]

    before = df.filter(pl.col(datetime_col) < cutoff)
    after = df.filter(pl.col(datetime_col) >= cutoff)

    return before, after


def main():
    """Run VG1 ES1 analysis."""
    df = read_data()
    print(df.describe())
    plot_histogram(df, "VG1_ES1_unprocessed_hist")
    plot_time_series(df, "VG1_ES1_unprocessed_time_evolution")

    print("\n ===Full Head===")
    print(df.head())

    bdf, adf = split_on_date(df, datetime(2020, 5, 1))
    print("\n ===BEFORE CUTOFF===")
    print(bdf.describe())

    print("\n ===AFTER CUTOFF===")
    print(adf.describe())


if __name__ == "__main__":
    main()
