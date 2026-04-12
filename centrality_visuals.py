from __future__ import annotations

from pathlib import Path
import warnings
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

try:
    import community as community_louvain
except Exception:  # pragma: no cover
    community_louvain = None

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_clean" / "final_migration_dataset.csv"
OUT = BASE / "deliverables_outputs" / "visuals"
NE_ZIP = OUT / "ne_110m_admin_0_countries.zip"
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 2024
TOP_CORRIDORS = 60


def clean_country(name: str) -> str:
    if not isinstance(name, str):
        return name
    return name.replace("*", "").strip()


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    edges = df.groupby(["origin", "destination"], as_index=False)["migrants"].sum()
    for _, row in edges.iterrows():
        if row["origin"] == row["destination"]:
            continue
        g.add_edge(row["origin"], row["destination"], weight=float(row["migrants"]))
    return g


def compute_centrality(g: nx.DiGraph) -> pd.DataFrame:
    degree = nx.degree_centrality(g)
    betw = nx.betweenness_centrality(g, weight="weight", normalized=True)
    close = nx.closeness_centrality(g)
    pr = nx.pagerank(g, weight="weight")

    return pd.DataFrame(
        {
            "country": list(g.nodes),
            "degree": [degree.get(n, 0.0) for n in g.nodes],
            "betweenness": [betw.get(n, 0.0) for n in g.nodes],
            "closeness": [close.get(n, 0.0) for n in g.nodes],
            "pagerank": [pr.get(n, 0.0) for n in g.nodes],
        }
    )


def assign_communities(g: nx.Graph) -> dict[str, int]:
    if community_louvain is not None:
        return community_louvain.best_partition(g.to_undirected(), weight="weight", random_state=42)
    # fallback: greedy modularity
    communities = list(nx.algorithms.community.greedy_modularity_communities(g.to_undirected(), weight="weight"))
    partition = {}
    for i, group in enumerate(sorted(communities, key=lambda c: (-len(c), sorted(c)[0]))):
        for node in group:
            partition[node] = i
    return partition


def prepare_geo(centrality: pd.DataFrame) -> pd.DataFrame:
    if gpd is None:
        raise RuntimeError("geopandas is required for map visuals")

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    if not NE_ZIP.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        NE_ZIP.write_bytes(resp.content)
    world = gpd.read_file(NE_ZIP)
    name_col = "NAME" if "NAME" in world.columns else world.columns[0]
    world["country_name"] = world[name_col]
    world["name_clean"] = world[name_col].apply(clean_country)

    c = centrality.copy()
    c["country_clean"] = c["country"].apply(clean_country)

    merged = world.merge(c, left_on="name_clean", right_on="country_clean", how="left")
    merged = merged.dropna(subset=["degree", "betweenness", "closeness", "pagerank"], how="all")
    merged = merged.to_crs(epsg=4326)
    merged["centroid"] = merged.geometry.centroid
    merged["lat"] = merged["centroid"].y
    merged["lon"] = merged["centroid"].x
    return merged


def plot_network(g: nx.DiGraph, centrality: pd.DataFrame, partition: dict[str, int]) -> None:
    size = centrality.set_index("country")["pagerank"].to_dict()
    colors = partition

    pos = nx.spring_layout(g, seed=42, k=0.25)
    nodes = list(g.nodes)
    node_sizes = [2000 * size.get(n, 0.0) + 50 for n in nodes]
    node_colors = [colors.get(n, 0) for n in nodes]

    plt.figure(figsize=(12, 10))
    edges = sorted(g.edges(data=True), key=lambda x: x[2].get("weight", 0), reverse=True)[:400]
    h = nx.DiGraph()
    h.add_edges_from(edges)
    nx.draw_networkx_edges(h, pos, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(g, pos, node_size=node_sizes, node_color=node_colors, cmap="tab20", alpha=0.85)
    plt.title("Migration Network (2024) — Node size by PageRank, Color by Community")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "network_pagerank_community.png", dpi=250)
    plt.close()


def plot_centrality_choropleth(geo: pd.DataFrame, column: str, title: str, out_file: str) -> None:
    fig = px.choropleth(
        geo,
        geojson=geo.geometry,
        locations=geo.index,
        color=column,
        projection="natural earth",
        hover_name="country_name",
        color_continuous_scale="Viridis",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=title, margin={"r":0,"t":40,"l":0,"b":0})
    fig.write_html(OUT / f"{out_file}.html")
    fig.write_image(OUT / f"{out_file}.png", scale=2)


def plot_flow_map(geo: pd.DataFrame, corridors: pd.DataFrame) -> None:
    # Build lookup for centroids
    coord = geo.set_index("name_clean")[["lat", "lon"]].to_dict("index")

    lines_lat = []
    lines_lon = []
    for _, row in corridors.iterrows():
        o = clean_country(row["origin"])
        d = clean_country(row["destination"])
        if o not in coord or d not in coord:
            continue
        lines_lat += [coord[o]["lat"], coord[d]["lat"], None]
        lines_lon += [coord[o]["lon"], coord[d]["lon"], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=lines_lat,
            lon=lines_lon,
            mode="lines",
            line=dict(width=0.7, color="#1f77b4"),
            opacity=0.5,
            name="Top corridors",
        )
    )

    fig.update_layout(
        title="Major Migration Corridors (Top 60, 2024)",
        geo=dict(showland=True, landcolor="rgb(240,240,240)")
    )
    fig.write_html(OUT / "migration_corridors_map.html")
    fig.write_image(OUT / "migration_corridors_map.png", scale=2)


def main() -> None:
    df = pd.read_csv(DATA)
    df["origin"] = df["origin"].apply(clean_country)
    df["destination"] = df["destination"].apply(clean_country)

    year_df = df[df["year"] == YEAR]
    g = build_graph(year_df)
    centrality = compute_centrality(g)
    partition = assign_communities(g)

    centrality.to_csv(OUT / f"centrality_{YEAR}.csv", index=False)

    # Centrality histograms
    for col in ["degree", "betweenness", "closeness", "pagerank"]:
        plt.figure(figsize=(6, 4))
        plt.hist(centrality[col], bins=30, color="#1f77b4", alpha=0.8)
        plt.title(f"{col.title()} Distribution ({YEAR})")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUT / f"centrality_{col}_hist.png", dpi=200)
        plt.close()

    plot_network(g, centrality, partition)

    if gpd is not None:
        geo = prepare_geo(centrality)
        plot_centrality_choropleth(geo, "degree", "Degree Centrality (2024)", "map_degree")
        plot_centrality_choropleth(geo, "betweenness", "Betweenness Centrality (2024)", "map_betweenness")
        plot_centrality_choropleth(geo, "closeness", "Closeness Centrality (2024)", "map_closeness")
        plot_centrality_choropleth(geo, "pagerank", "PageRank (2024)", "map_pagerank")

        corridors = (
            year_df.groupby(["origin", "destination"], as_index=False)["migrants"]
            .sum()
            .sort_values("migrants", ascending=False)
            .head(TOP_CORRIDORS)
        )
        plot_flow_map(geo, corridors)

    print(f"Centrality visuals saved to {OUT}")


if __name__ == "__main__":
    main()
