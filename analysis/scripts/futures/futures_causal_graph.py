"""Plot a pre-post lockdown measures causal graph."""

from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import seaborn as sns
from futures_constants import MIN_TICKS_PER_DAY, NETWORK_PATH, RETURNS_DATA
from futures_main import date_return_cols, get_bootstrap_maximised_te
from wrangling.columns import Cols, Columns
from wrangling.data_builder import FuturesDataBuilder

sns.set()


def create_and_plot_te_graph(result, threshold=0.0, plot_path=""):
    """Create and plot TE graphs with minimal customization."""
    graph = nx.DiGraph()

    for edge, weight in result.items():
        if float(weight) > threshold:
            source, target = edge.split("->")
            graph.add_edge(source, target, weight=float(weight))

    pos = nx.spring_layout(graph, seed=42, k=0.8)

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_weight="bold",
        arrows=True,
    )
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={e: f"{w:.3f}" for e, w in labels.items()},
        label_pos=0.8,
    )
    plt.title("Pre-Lockdown Causal Graph")
    plt.axis("off")
    plt.savefig(plot_path, dpi=300)
    plt.close()


cols = Cols.get_all_instruments()
p1start = datetime(2019, 11, 1)
p1end = datetime(2019, 11, 8)
p2start = datetime(2020, 3, 15)
p2end = datetime(2020, 3, 22)


def max_bootstrap_early_stopping_closure(src, tgt, df):
    """Get bootstrap maximised TE with fixed params for early stopping.

    This was added to cross-check the early stopping checks.
    They were conclusive and the correct binnings were identified.
    """
    te, bs, _ft = get_bootstrap_maximised_te(
        src, tgt, df, window_size=30, trend_patience=30
    )
    return float(max((te - bs) / te, 0))


if __name__ == "__main__":
    print("Building data frame")
    futures = FuturesDataBuilder.load(RETURNS_DATA)
    df = (
        (
            futures.with_datetime_index()
            .drop_nulls()
            .log_returns()
            .drop_incomplete_trading_days(MIN_TICKS_PER_DAY)
        )
        .build()
        .select(date_return_cols)
    )

    pre_result = {}
    post_result = {}

    pre_df = df.filter(
        (pl.col(Columns.Date) >= p1start) & (pl.col(Columns.Date) <= p1end)
    )
    post_df = df.filter(
        (pl.col(Columns.Date) >= p2start) & (pl.col(Columns.Date) <= p2end)
    )

    print("Calculating transfer entropy for pre-lockdown period...")
    for src, tgt in [(s, t) for s in cols for t in cols if s != t]:
        pre_result[f"{src.base}->{tgt.base}"] = max_bootstrap_early_stopping_closure(
            src, tgt, post_df
        )

    print("Calculating transfer entropy for post-lockdown period...")
    for src, tgt in [(s, t) for s in cols for t in cols if s != t]:
        post_result[f"{src.base}->{tgt.base}"] = max_bootstrap_early_stopping_closure(
            src, tgt, post_df
        )

    create_and_plot_te_graph(
        pre_result, plot_path=NETWORK_PATH / f"pre-lockdown_{p1start}-{p1end}.png"
    )
    create_and_plot_te_graph(
        post_result, plot_path=NETWORK_PATH / f"in-lockdown-{p2start}-{p2end}.png"
    )
