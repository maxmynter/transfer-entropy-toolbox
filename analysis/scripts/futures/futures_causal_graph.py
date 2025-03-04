"""Plot a pre-post lockdown measures causal graph."""

import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
import seaborn as sns
from futures_constants import DATA_PATH, NETWORK_PATH
from futures_main import config, filename, get_bootstrap_maximised_te
from wrangling.columns import Cols

sns.set()

NODE_POSITIONS = None


def create_and_plot_te_graph(result: dict, threshold=0.0, plot_path=""):
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

    for edge, weight in result.items():
        if float(weight) > threshold:
            src, tgt = edge.split("->")
            graph.add_edge(src, tgt, weight=float(weight))
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


csv_filename = filename + ".csv"

if __name__ == "__main__":
    cols = Cols.get_all_instruments()
    df = pl.read_csv(DATA_PATH / csv_filename)
    df = df[config.WINDOW_SIZE :]
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
        )
