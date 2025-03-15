"""Plot a pre-post lockdown measures causal graph."""

from datetime import date, datetime

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import seaborn as sns
from futures_constants import DATA_PATH, NETWORK_PATH
from futures_main import config, filename, get_bootstrap_maximised_te
from wrangling.columns import Cols

StatsRecord = dict[str, str | datetime | date | float | int]
StatsTracker = list[StatsRecord]
sns.set()

NODE_POSITIONS = None


def create_and_plot_te_graph(
    result: dict, threshold=0.0, plot_path="", stats_tracker=None, current_date=None
):
    """Plot graph for result."""
    global NODE_POSITIONS  # noqa: PLW0603

    graph = nx.DiGraph()

    all_instruments = set()

    for edge, _ in result.items():
        src, tgt = edge.split("->")
        all_instruments.add(src)
        all_instruments.add(tgt)

    for instrument in all_instruments:
        graph.add_node(instrument)

    weights = []
    for edge, weight in result.items():
        if float(weight) > threshold:
            src, tgt = edge.split("->")
            graph.add_edge(src, tgt, weight=float(weight))
            weights.append(float(weight))

    avg_weight = sum(weights) / len(weights) if weights else 0
    median_weight = sorted(weights)[len(weights) // 2] if weights else 0

    if stats_tracker is not None and current_date is not None:
        stats_tracker.append(
            {
                "date": current_date,
                "average": avg_weight,
                "median": median_weight,
                "count": len(weights),
            }
        )

    if NODE_POSITIONS is None:
        NODE_POSITIONS = nx.spring_layout(graph, seed=42, k=0.8)

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        NODE_POSITIONS,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_weight="bold",
        arrows=True,
    )

    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph,
        NODE_POSITIONS,
        edge_labels={e: f"{w:.3f}" for e, w in labels.items()},
        label_pos=0.8,
    )

    plt.text(
        0.05,
        0.05,  # Position in figure coordinates (bottom left)
        f"Average: {avg_weight:.3f}\nMedian: {median_weight:.3f}",
        transform=plt.gcf().transFigure,  # Use figure coordinates
        bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round"},
        fontsize=12,
    )

    plt.axis("off")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def max_bootstrap_early_stopping_closure(src, tgt, df):
    """Get bootstrap maximised TE with fixed params for early stopping.

    This was added to cross-check the early stopping checks.
    They were conclusive and the correct binnings were identified.
    """
    te, bs, _ft = get_bootstrap_maximised_te(
        src, tgt, df, window_size=20, trend_patience=20
    )
    return te


def plot_statistics_over_time(stats_tracker, plot_path):
    """Create a plot of network statistics over time using Polars."""
    if not stats_tracker:
        return

    # Convert date strings to datetime objects if needed
    stats_df = pl.DataFrame(stats_tracker)

    # Make sure date is in datetime format
    if isinstance(stats_df["date"][0], str):
        stats_df = stats_df.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

    # Sort by date
    stats_df = stats_df.sort("date")

    # Convert to lists for plotting
    dates = stats_df["date"].to_list()
    averages = stats_df["average"].to_list()
    medians = stats_df["median"].to_list()

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot average and median
    ax1.plot(dates, averages, "b-", label="Average Transfer Entropy")
    ax1.plot(dates, medians, "r-", label="Median Transfer Entropy")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Transfer Entropy Value", color="b")

    # Formatting
    plt.title("Network Statistics Over Time")

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper left")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


csv_filename = filename + ".csv"

if __name__ == "__main__":
    cols = Cols.get_all_instruments()
    df = pl.read_csv(DATA_PATH / csv_filename)
    df = df[config.WINDOW_SIZE :]

    stats_tracker: StatsTracker = []
    for current_date in df[Cols.Date].to_list():
        print(f"Processing window: {config.WINDOW_SIZE} days to {current_date}")

        result = {}
        pairs = [(s, t) for s in cols for t in cols if s != t]
        for src, tgt in pairs:
            result[f"{src.base}->{tgt.base}"] = (
                df.filter(pl.col(Cols.Date) == current_date)
                .select(f"{src.base}->{tgt.base}")
                .item()
            )
        create_and_plot_te_graph(
            result,
            threshold=0.0,
            plot_path=NETWORK_PATH / f"network_{config.WINDOW_SIZE}_days_to"
            f"_{current_date}.png",
            stats_tracker=stats_tracker,
            current_date=current_date,
        )

    plot_statistics_over_time(
        stats_tracker,
        plot_path=NETWORK_PATH
        / f"network_statistics_{config.WINDOW_SIZE}_days_to_{current_date}.png",
    )
