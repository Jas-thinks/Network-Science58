from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    import community as community_louvain
except Exception:  # pragma: no cover - optional dependency
    community_louvain = None

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_clean" / "final_migration_dataset.csv"
OUT = BASE / "analysis_outputs"
OUT.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_eda(df: pd.DataFrame) -> None:
    (OUT / "tables").mkdir(exist_ok=True)

    # shape and columns
    shape_df = pd.DataFrame({"rows": [len(df)], "columns": [len(df.columns)]})
    shape_df.to_csv(OUT / "tables" / "shape.csv", index=False)

    pd.DataFrame({"columns": df.columns}).to_csv(OUT / "tables" / "columns.csv", index=False)

    # summary stats
    df.describe(include="all").to_csv(OUT / "tables" / "summary_stats.csv")

    # top corridors
    top_corridors = (
        df.groupby(["origin", "destination"], as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )
    top_corridors.to_csv(OUT / "tables" / "top_corridors.csv", index=False)

    # top destinations (inflow)
    top_dest = (
        df.groupby("destination", as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )
    top_dest.to_csv(OUT / "tables" / "top_destinations.csv", index=False)

    # top origins (outflow)
    top_origin = (
        df.groupby("origin", as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )
    top_origin.to_csv(OUT / "tables" / "top_origins.csv", index=False)

    # correlation heatmap
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="viridis", linewidths=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=200)
    plt.close()


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    edges = df.groupby(["origin", "destination"], as_index=False)["migrants"].sum()
    for _, row in edges.iterrows():
        g.add_edge(row["origin"], row["destination"], weight=row["migrants"])
    return g


def centrality_analysis(g: nx.DiGraph) -> pd.DataFrame:
    indeg = nx.in_degree_centrality(g)
    outdeg = nx.out_degree_centrality(g)
    betw = nx.betweenness_centrality(g, weight="weight", normalized=True)
    pr = nx.pagerank(g, weight="weight")

    centrality = pd.DataFrame({
        "country": list(g.nodes),
        "in_degree": [indeg[n] for n in g.nodes],
        "out_degree": [outdeg[n] for n in g.nodes],
        "betweenness": [betw[n] for n in g.nodes],
        "pagerank": [pr[n] for n in g.nodes],
    })

    centrality.to_csv(OUT / "tables" / "centrality.csv", index=False)
    return centrality


def community_detection(g: nx.DiGraph) -> None:
    if community_louvain is None:
        return

    undirected = g.to_undirected()
    partition = community_louvain.best_partition(undirected, weight="weight")
    modularity = community_louvain.modularity(partition, undirected, weight="weight")

    comm_df = pd.DataFrame({"country": list(partition.keys()), "community": list(partition.values())})
    comm_df.to_csv(OUT / "tables" / "communities.csv", index=False)

    pd.DataFrame({"modularity": [modularity]}).to_csv(OUT / "tables" / "modularity.csv", index=False)


def temporal_analysis(df: pd.DataFrame) -> None:
    if "year" not in df.columns:
        return

    (OUT / "temporal").mkdir(exist_ok=True)

    hubs = []
    for year, sub in df.groupby("year"):
        g = build_graph(sub)
        indeg = nx.in_degree_centrality(g)
        outdeg = nx.out_degree_centrality(g)
        pr = nx.pagerank(g, weight="weight")

        top_in = sorted(indeg.items(), key=lambda x: x[1], reverse=True)[:10]
        top_out = sorted(outdeg.items(), key=lambda x: x[1], reverse=True)[:10]
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]

        for rank, (c, v) in enumerate(top_in, 1):
            hubs.append({"year": year, "metric": "in_degree", "rank": rank, "country": c, "value": v})
        for rank, (c, v) in enumerate(top_out, 1):
            hubs.append({"year": year, "metric": "out_degree", "rank": rank, "country": c, "value": v})
        for rank, (c, v) in enumerate(top_pr, 1):
            hubs.append({"year": year, "metric": "pagerank", "rank": rank, "country": c, "value": v})

    pd.DataFrame(hubs).to_csv(OUT / "temporal" / "top_hubs_over_time.csv", index=False)


def visualize_network(g: nx.DiGraph) -> None:
    # Degree distribution
    degrees = [d for _, d in g.degree()]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUT / "degree_distribution.png", dpi=200)
    plt.close()

    # Simplified network (top edges only)
    edges = sorted(g.edges(data=True), key=lambda e: e[2].get("weight", 0), reverse=True)[:300]
    h = nx.DiGraph()
    h.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(h, seed=42, k=0.3)
    nx.draw_networkx_nodes(h, pos, node_size=30, alpha=0.7)
    nx.draw_networkx_edges(h, pos, alpha=0.2, width=0.5)
    plt.title("Simplified Migration Network (Top 300 Edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "network_graph_simplified.png", dpi=200)
    plt.close()


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing input file: {DATA}")

    df = load_data(DATA)

    basic_eda(df)

    g = build_graph(df)
    centrality_analysis(g)
    community_detection(g)
    temporal_analysis(df)
    visualize_network(g)

    print(f"Analysis outputs saved to {OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
