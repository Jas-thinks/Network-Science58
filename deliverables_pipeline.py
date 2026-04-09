from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

try:
    import community as community_louvain
except Exception:  # pragma: no cover - optional dependency
    community_louvain = None

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:  # pragma: no cover - optional dependency
    variance_inflation_factor = None

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_clean" / "final_migration_dataset.csv"
OUT = BASE / "deliverables_outputs"
SNAPSHOTS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]


@dataclass
class SnapshotResult:
    year: int
    graph: nx.DiGraph
    centrality: pd.DataFrame
    partition: dict[str, int] | None
    modularity: float | None


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": OUT,
        "graphs": OUT / "dynamic_graphs",
        "centrality": OUT / "centrality_csvs",
        "community": OUT / "community_detection",
        "regression": OUT / "regression_output",
        "dashboard": OUT / "powerbi_exports",
        "tables": OUT / "tables",
        "screenshots": OUT / "powerbi_exports" / "screenshots",
        "data": OUT / "powerbi_exports" / "data",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing input file: {DATA}")
    df = pd.read_csv(DATA)
    expected = {"destination", "origin", "year", "migrants"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset missing columns: {sorted(missing)}")
    return df


def build_snapshot_graph(df: pd.DataFrame, year: int) -> nx.DiGraph:
    sub = df[df["year"] == year].copy()
    g = nx.DiGraph(year=year)
    edges = sub.groupby(["origin", "destination"], as_index=False)["migrants"].sum()
    for _, row in edges.iterrows():
        g.add_edge(
            row["origin"],
            row["destination"],
            weight=float(row["migrants"]),
        )
    return g


def compute_centrality(g: nx.DiGraph, year: int) -> pd.DataFrame:
    indeg = nx.in_degree_centrality(g)
    outdeg = nx.out_degree_centrality(g)
    pr = nx.pagerank(g, weight="weight") if g.number_of_nodes() > 0 else {}
    if g.number_of_nodes() > 1:
        if g.number_of_nodes() <= 200:
            betw = nx.betweenness_centrality(g, weight="weight", normalized=True)
        else:
            betw = nx.betweenness_centrality(
                g,
                k=min(75, g.number_of_nodes()),
                weight="weight",
                normalized=True,
                seed=42,
            )
    else:
        betw = {n: 0.0 for n in g.nodes}

    df = pd.DataFrame(
        {
            "year": year,
            "country": list(g.nodes),
            "in_degree": [indeg.get(n, 0.0) for n in g.nodes],
            "out_degree": [outdeg.get(n, 0.0) for n in g.nodes],
            "betweenness": [betw.get(n, 0.0) for n in g.nodes],
            "pagerank": [pr.get(n, 0.0) for n in g.nodes],
            "degree": [g.degree(n) for n in g.nodes],
            "in_strength": [g.in_degree(n, weight="weight") for n in g.nodes],
            "out_strength": [g.out_degree(n, weight="weight") for n in g.nodes],
        }
    )
    return df.sort_values(["pagerank", "out_strength"], ascending=False)


def community_partition(g: nx.DiGraph) -> tuple[dict[str, int] | None, float | None]:
    if g.number_of_nodes() == 0:
        return None, None
    undirected = g.to_undirected()
    if community_louvain is not None:
        partition = community_louvain.best_partition(undirected, weight="weight", random_state=42)
        modularity = community_louvain.modularity(partition, undirected, weight="weight")
        return partition, float(modularity)
    # Fallback: greedy modularity communities with deterministic ordering.
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    partition: dict[str, int] = {}
    for idx, community in enumerate(sorted(communities, key=lambda c: (-len(c), sorted(c)[0]))):
        for node in community:
            partition[node] = idx
    modularity = nx.algorithms.community.quality.modularity(
        undirected, communities, weight="weight"
    )
    return partition, float(modularity)


def community_sets(partition: dict[str, int]) -> dict[int, set[str]]:
    groups: dict[int, set[str]] = defaultdict(set)
    for node, comm in partition.items():
        groups[int(comm)].add(node)
    return dict(groups)


def best_match_jaccard(part_a: dict[int, set[str]], part_b: dict[int, set[str]]) -> tuple[float, float, pd.DataFrame]:
    rows = []
    scores = []
    for comm_a, nodes_a in part_a.items():
        best = (None, 0.0, 0, 0)
        for comm_b, nodes_b in part_b.items():
            inter = len(nodes_a & nodes_b)
            union = len(nodes_a | nodes_b)
            score = inter / union if union else 0.0
            if score > best[1]:
                best = (comm_b, score, inter, union)
        rows.append(
            {
                "community_a": comm_a,
                "community_b": best[0],
                "jaccard": best[1],
                "intersection": best[2],
                "union": best[3],
                "size_a": len(nodes_a),
                "matched_size_b": len(part_b.get(best[0], set())) if best[0] is not None else 0,
            }
        )
        scores.append(best[1])
    summary_mean = float(np.mean(scores)) if scores else 0.0
    summary_median = float(np.median(scores)) if scores else 0.0
    return summary_mean, summary_median, pd.DataFrame(rows)


def nmi_ari_between(part_a: dict[str, int], part_b: dict[str, int]) -> tuple[float, float, int]:
    nodes = sorted(set(part_a) & set(part_b))
    if not nodes:
        return 0.0, 0.0, 0
    labels_a = [part_a[n] for n in nodes]
    labels_b = [part_b[n] for n in nodes]
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    ari = adjusted_rand_score(labels_a, labels_b)
    return float(nmi), float(ari), len(nodes)


def write_graph(g: nx.DiGraph, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(g, f)


def save_snapshot_outputs(df: pd.DataFrame, paths: dict[str, Path]) -> list[SnapshotResult]:
    results: list[SnapshotResult] = []
    all_cent = []
    for year in SNAPSHOTS:
        g = build_snapshot_graph(df, year)
        cent = compute_centrality(g, year)
        partition, modularity = community_partition(g)
        results.append(SnapshotResult(year, g, cent, partition, modularity))

        graph_path = paths["graphs"] / f"dynamic_digraph_{year}.pkl"
        write_graph(g, graph_path)
        cent.to_csv(paths["centrality"] / f"centrality_{year}.csv", index=False)
        all_cent.append(cent)

    combined = pd.concat(all_cent, ignore_index=True)
    combined.to_csv(paths["centrality"] / "centrality_all_snapshots.csv", index=False)
    return results


def save_community_outputs(results: list[SnapshotResult], paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modularity_rows = []
    similarity_rows = []
    drift_summary_rows = []
    drift_detail_frames = []

    for res in results:
        partition = res.partition or {}
        communities = community_sets(partition) if partition else {}
        modularity_rows.append(
            {
                "year": res.year,
                "nodes": res.graph.number_of_nodes(),
                "edges": res.graph.number_of_edges(),
                "communities": len(set(partition.values())) if partition else 0,
                "modularity_Q": res.modularity if res.modularity is not None else np.nan,
            }
        )
        if partition:
            comm_df = pd.DataFrame(
                sorted(((node, comm) for node, comm in partition.items()), key=lambda x: (x[1], x[0])),
                columns=["country", "community"],
            )
            comm_df.insert(0, "year", res.year)
            comm_df.to_csv(paths["community"] / f"communities_{res.year}.csv", index=False)

    for prev, curr in zip(results[:-1], results[1:]):
        nmi, ari, common_nodes = nmi_ari_between(prev.partition or {}, curr.partition or {})
        similarity_rows.append(
            {
                "from_year": prev.year,
                "to_year": curr.year,
                "common_nodes": common_nodes,
                "nmi": nmi,
                "ari": ari,
            }
        )
        prev_sets = community_sets(prev.partition or {})
        curr_sets = community_sets(curr.partition or {})
        mean_j, med_j, detail = best_match_jaccard(prev_sets, curr_sets)
        drift_summary_rows.append(
            {
                "from_year": prev.year,
                "to_year": curr.year,
                "mean_best_jaccard": mean_j,
                "median_best_jaccard": med_j,
                "communities_compared": len(prev_sets),
            }
        )
        detail.insert(0, "to_year", curr.year)
        detail.insert(0, "from_year", prev.year)
        drift_detail_frames.append(detail)

    modularity_df = pd.DataFrame(modularity_rows)
    similarity_df = pd.DataFrame(similarity_rows)
    drift_summary_df = pd.DataFrame(drift_summary_rows)
    drift_detail_df = pd.concat(drift_detail_frames, ignore_index=True) if drift_detail_frames else pd.DataFrame()

    modularity_df.to_csv(paths["community"] / "modularity_Q_table.csv", index=False)
    similarity_df.to_csv(paths["community"] / "nmi_ari_comparison.csv", index=False)
    drift_summary_df.to_csv(paths["community"] / "jaccard_drift_summary.csv", index=False)
    drift_detail_df.to_csv(paths["community"] / "jaccard_drift_detail.csv", index=False)

    return modularity_df, similarity_df, drift_summary_df


def plot_community_outputs(modularity_df: pd.DataFrame, similarity_df: pd.DataFrame, drift_summary_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    sns.set_theme(style="whitegrid")
    # Modularity + similarity overview
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    axes[0].plot(modularity_df["year"], modularity_df["modularity_Q"], marker="o", color="#1f77b4")
    axes[0].set_title("Modularity Q by Snapshot")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Q")

    axes[1].plot(similarity_df["to_year"], similarity_df["nmi"], marker="o", label="NMI", color="#2ca02c")
    axes[1].plot(similarity_df["to_year"], similarity_df["ari"], marker="o", label="ARI", color="#d62728")
    axes[1].set_title("Community Stability Between Consecutive Snapshots")
    axes[1].set_xlabel("Transition To Year")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    axes[2].bar(
        drift_summary_df["to_year"].astype(str),
        drift_summary_df["mean_best_jaccard"],
        color="#9467bd",
    )
    axes[2].set_title("Mean Best-Match Jaccard Drift")
    axes[2].set_xlabel("Snapshot Transition To Year")
    axes[2].set_ylabel("Jaccard")
    fig.savefig(paths["community"] / "community_stability_overview.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    plt.plot(drift_summary_df["to_year"], drift_summary_df["mean_best_jaccard"], marker="o", linewidth=2)
    plt.title("Jaccard Drift Across Community Partitions")
    plt.xlabel("Year")
    plt.ylabel("Mean best-match Jaccard")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(paths["community"] / "jaccard_drift_chart.png", dpi=200)
    plt.close()


def build_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["origin", "destination", "year"]).copy()

    # Lag-style historical features built from prior years only.
    out["pair_prev_total"] = (
        out.groupby(["origin", "destination"])["migrants"].cumsum().shift(1)
    )
    out["origin_prev_total"] = out.groupby("origin")["migrants"].cumsum().shift(1)
    out["destination_prev_total"] = out.groupby("destination")["migrants"].cumsum().shift(1)
    out["pair_prev_count"] = out.groupby(["origin", "destination"]).cumcount()
    out["origin_prev_count"] = out.groupby("origin").cumcount()
    out["destination_prev_count"] = out.groupby("destination").cumcount()
    out["code_gap"] = (out["dest_code"] - out["orig_code"]).abs()
    out["year_index"] = out["year"] - out["year"].min()

    # Fill first-observation gaps with zero to preserve rows.
    for col in [
        "pair_prev_total",
        "origin_prev_total",
        "destination_prev_total",
    ]:
        out[col] = out[col].fillna(0.0)

    # Log target stabilizes heavy-tailed migration counts.
    out["log_migrants"] = np.log1p(out["migrants"])
    return out


def select_regression_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [
        "year_index",
        "dest_code",
        "orig_code",
        "code_gap",
        "pair_prev_total",
        "origin_prev_total",
        "destination_prev_total",
        "pair_prev_count",
        "origin_prev_count",
        "destination_prev_count",
    ]
    X = df[feature_cols].copy()
    y = df["log_migrants"].copy()
    return X, y, feature_cols


def train_regression_models(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = X["year_index"] <= 25  # up to 2015; test on 2020 and 2024
    X_train = X.loc[train_mask].copy()
    X_test = X.loc[~train_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[~train_mask].copy()

    numeric_cols = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )

    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.001, max_iter=20000, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    results = []
    coef_tables = []
    fitted_preprocessor = None
    fitted_models = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2 = float(r2_score(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results.append({"model": name, "r2": r2, "rmse": rmse, "test_rows": len(y_test)})
        fitted_models[name] = pipe
        if name in {"linear", "ridge", "lasso"}:
            coefs = pipe.named_steps["model"].coef_
            coef_tables.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "feature": numeric_cols,
                        "coefficient": coefs,
                        "abs_coefficient": np.abs(coefs),
                    }
                ).sort_values("abs_coefficient", ascending=False)
            )
        fitted_preprocessor = pipe.named_steps["preprocessor"]

    result_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    coef_df = pd.concat(coef_tables, ignore_index=True)

    # VIF on scaled, imputed training predictors for stable diagnostics.
    train_matrix = pd.DataFrame(
        fitted_preprocessor.fit_transform(X_train),
        columns=numeric_cols,
        index=X_train.index,
    )
    vif_rows = []
    if variance_inflation_factor is not None:
        for i, col in enumerate(train_matrix.columns):
            vif_rows.append({"feature": col, "VIF": float(variance_inflation_factor(train_matrix.values, i))})
    else:
        vif_rows = [{"feature": col, "VIF": np.nan} for col in train_matrix.columns]
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

    return result_df, coef_df, vif_df


def save_regression_outputs(result_df: pd.DataFrame, coef_df: pd.DataFrame, vif_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    result_df.to_csv(paths["regression"] / "r2_comparison.csv", index=False)
    coef_df.to_csv(paths["regression"] / "coefficient_tables.csv", index=False)
    vif_df.to_csv(paths["regression"] / "vif_report.csv", index=False)

    for model_name in coef_df["model"].unique():
        coef_df.loc[coef_df["model"] == model_name].to_csv(
            paths["regression"] / f"coefficients_{model_name}.csv", index=False
        )

    # Model comparison chart.
    plt.figure(figsize=(8, 4))
    sns.barplot(data=result_df, x="model", y="r2", palette="viridis")
    plt.title("Regression R² Comparison")
    plt.xlabel("Model")
    plt.ylabel("R² on test set")
    plt.tight_layout()
    plt.savefig(paths["regression"] / "r2_comparison.png", dpi=200)
    plt.close()

    # VIF chart.
    plt.figure(figsize=(10, 5))
    sns.barplot(data=vif_df.sort_values("VIF", ascending=False), x="VIF", y="feature", color="#ff7f0e")
    plt.title("Variance Inflation Factor (VIF)")
    plt.xlabel("VIF")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(paths["regression"] / "vif_report.png", dpi=200)
    plt.close()


def build_dashboard_data(df: pd.DataFrame, results: list[SnapshotResult], modularity_df: pd.DataFrame, similarity_df: pd.DataFrame, drift_summary_df: pd.DataFrame, result_df: pd.DataFrame, coef_df: pd.DataFrame, vif_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    latest_year = SNAPSHOTS[-1]
    latest = df[df["year"] == latest_year].copy()

    overview = (
        latest.groupby("destination", as_index=False)["migrants"].sum().sort_values("migrants", ascending=False).head(15)
    )
    overview.to_csv(paths["data"] / "dashboard1_overview.csv", index=False)

    corridors = (
        latest.groupby(["origin", "destination"], as_index=False)["migrants"].sum().sort_values("migrants", ascending=False).head(20)
    )
    corridors.to_csv(paths["data"] / "dashboard2_corridors.csv", index=False)

    temporal = (
        df.groupby("year", as_index=False)["migrants"].sum().sort_values("year")
    )
    temporal.to_csv(paths["data"] / "dashboard3_temporal.csv", index=False)

    communities = modularity_df.copy()
    communities.to_csv(paths["data"] / "dashboard4_communities.csv", index=False)

    regression = result_df.copy()
    regression.to_csv(paths["data"] / "dashboard5_regression.csv", index=False)

    # 1. Executive overview dashboard.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sns.barplot(data=overview, y="destination", x="migrants", ax=axes[0, 0], color="#1f77b4")
    axes[0, 0].set_title("Top Destinations")
    axes[0, 0].set_xlabel("Migrants")
    axes[0, 0].set_ylabel("")
    sns.lineplot(data=temporal, x="year", y="migrants", marker="o", ax=axes[0, 1], color="#2ca02c")
    axes[0, 1].set_title("Total Migrants by Snapshot")
    sns.barplot(data=modularity_df, x="year", y="modularity_Q", ax=axes[1, 0], color="#9467bd")
    axes[1, 0].set_title("Community Modularity")
    sns.barplot(data=result_df, x="model", y="r2", ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("Regression R²")
    fig.savefig(paths["screenshots"] / "dashboard_01_overview.png", dpi=200)
    plt.close(fig)

    # 2. Corridor dashboard.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    top_corridors = corridors.copy()
    top_corridors["corridor"] = top_corridors["origin"] + " → " + top_corridors["destination"]
    sns.barplot(data=top_corridors, y="corridor", x="migrants", ax=axes[0], color="#ff7f0e")
    axes[0].set_title("Top Migration Corridors")
    axes[0].set_xlabel("Migrants")
    axes[0].set_ylabel("")
    axes[1].pie(
        [top_corridors["migrants"].sum(), latest["migrants"].sum() - top_corridors["migrants"].sum()],
        labels=["Top corridors", "Other flows"],
        autopct="%1.1f%%",
        colors=["#ffbb78", "#d9d9d9"],
    )
    axes[1].set_title("Flow Concentration")
    fig.savefig(paths["screenshots"] / "dashboard_02_corridors.png", dpi=200)
    plt.close(fig)

    # 3. Temporal dashboard.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sns.lineplot(data=temporal, x="year", y="migrants", marker="o", ax=axes[0, 0], color="#1f77b4")
    axes[0, 0].set_title("Total Migrants")
    sns.lineplot(data=modularity_df, x="year", y="modularity_Q", marker="o", ax=axes[0, 1], color="#9467bd")
    axes[0, 1].set_title("Modularity Trend")
    sns.lineplot(data=similarity_df, x="to_year", y="nmi", marker="o", ax=axes[1, 0], label="NMI")
    sns.lineplot(data=similarity_df, x="to_year", y="ari", marker="o", ax=axes[1, 0], label="ARI")
    axes[1, 0].set_title("Community Similarity")
    axes[1, 0].legend()
    sns.lineplot(data=drift_summary_df, x="to_year", y="mean_best_jaccard", marker="o", ax=axes[1, 1], color="#8c564b")
    axes[1, 1].set_title("Jaccard Drift")
    fig.savefig(paths["screenshots"] / "dashboard_03_temporal.png", dpi=200)
    plt.close(fig)

    # 4. Community dashboard.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    sns.barplot(data=modularity_df, x="year", y="communities", ax=axes[0], color="#2ca02c")
    axes[0].set_title("Detected Communities")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count")
    sns.barplot(data=modularity_df, x="year", y="modularity_Q", ax=axes[1], color="#9467bd")
    axes[1].set_title("Modularity Q")
    fig.savefig(paths["screenshots"] / "dashboard_04_communities.png", dpi=200)
    plt.close(fig)

    # 5. Regression diagnostics dashboard.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    sns.barplot(data=result_df, x="model", y="r2", ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("R² Comparison")
    top_coefs = coef_df.loc[coef_df["model"] == "linear"].head(8).copy()
    sns.barplot(data=top_coefs, y="feature", x="coefficient", ax=axes[0, 1], color="#1f77b4")
    axes[0, 1].set_title("Linear Coefficients")
    sns.barplot(data=vif_df.head(8), y="feature", x="VIF", ax=axes[1, 0], color="#ff7f0e")
    axes[1, 0].set_title("Top VIF Features")
    sns.lineplot(data=temporal, x="year", y="migrants", marker="o", ax=axes[1, 1], color="#2ca02c")
    axes[1, 1].set_title("Flow Trend")
    fig.savefig(paths["screenshots"] / "dashboard_05_regression.png", dpi=200)
    plt.close(fig)


def write_final_report(modularity_df: pd.DataFrame, similarity_df: pd.DataFrame, drift_summary_df: pd.DataFrame, result_df: pd.DataFrame, coef_df: pd.DataFrame, vif_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    best_model = result_df.iloc[0]
    top_mod = modularity_df.sort_values("modularity_Q", ascending=False).iloc[0]
    report = f"""# Final Report: Global Migration Network Deliverables

## 1. Project Summary
This project converts the UN DESA bilateral migration stock data into a dynamic directed network, community structure diagnostics, and regression-ready analytical outputs.

## 2. Key Deliverables Produced
- 8-snapshot dynamic DiGraph files in `deliverables_outputs/dynamic_graphs/`
- Centrality CSVs in `deliverables_outputs/centrality_csvs/`
- Community detection tables, similarity comparisons, and drift charts in `deliverables_outputs/community_detection/`
- Regression coefficient tables, R² comparison, and VIF report in `deliverables_outputs/regression_output/`
- Five dashboard screenshot exports and dashboard-ready CSVs in `deliverables_outputs/powerbi_exports/`

## 3. Dynamic Network Summary
- Snapshots analysed: {len(modularity_df)}
- Highest modularity snapshot: {int(top_mod['year'])} with Q = {top_mod['modularity_Q']:.4f}
- Mean NMI across transitions: {similarity_df['nmi'].mean():.4f}
- Mean ARI across transitions: {similarity_df['ari'].mean():.4f}
- Mean best-match Jaccard drift: {drift_summary_df['mean_best_jaccard'].mean():.4f}

## 4. Regression Summary
- Best model: {best_model['model']}
- Best test R²: {best_model['r2']:.4f}
- Best test RMSE: {best_model['rmse']:.4f}
- Highest VIF feature: {vif_df.iloc[0]['feature']} ({vif_df.iloc[0]['VIF']:.4f})

## 5. Applications
1. **Policy targeting**: identify the most influential migration corridors and destination hubs for planning and humanitarian response.
2. **Regional integration**: monitor community stability to detect persistent migration blocs and cross-regional ties.
3. **Risk monitoring**: use centrality and community drift to flag corridor shocks and sudden redistribution of flows.
4. **Forecasting**: regression outputs offer a baseline for flow prediction and feature screening.
5. **Dashboarding**: the exported CSVs can be imported into Power BI to recreate interactive views.

## 6. Preliminary References
- United Nations, Department of Economic and Social Affairs. *International Migrant Stock 2024*.
- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- Newman, M. E. J. & Girvan, M. (2004). Finding and evaluating community structure in networks.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks.
- Fortunato, S. (2010). Community detection in graphs.
- Krzanowski, W. J. (1988). Principles of multivariate analysis: a user's perspective.

## 7. Notes on Power BI
This Linux workspace can export dashboard-ready data and screenshots, but it cannot author native `.pbix` files without Power BI Desktop. The generated CSVs and screenshot mocks are structured so they can be imported into Power BI with minimal setup.
"""
    (paths["root"] / "final_report.md").write_text(report, encoding="utf-8")

    # Also write a compact JSON summary for quick reuse.
    summary = {
        "best_model": best_model.to_dict(),
        "highest_modularity_snapshot": top_mod.to_dict(),
        "mean_nmi": float(similarity_df["nmi"].mean()),
        "mean_ari": float(similarity_df["ari"].mean()),
        "mean_jaccard_drift": float(drift_summary_df["mean_best_jaccard"].mean()),
    }
    (paths["root"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    paths = ensure_dirs()
    df = load_data()
    results = save_snapshot_outputs(df, paths)
    modularity_df, similarity_df, drift_summary_df = save_community_outputs(results, paths)
    plot_community_outputs(modularity_df, similarity_df, drift_summary_df, paths)

    reg_df = build_regression_features(df)
    X, y, _ = select_regression_columns(reg_df)
    result_df, coef_df, vif_df = train_regression_models(X, y)
    save_regression_outputs(result_df, coef_df, vif_df, paths)

    build_dashboard_data(df, results, modularity_df, similarity_df, drift_summary_df, result_df, coef_df, vif_df, paths)
    write_final_report(modularity_df, similarity_df, drift_summary_df, result_df, coef_df, vif_df, paths)

    print(f"Deliverables written to {paths['root']}")
    print("Top regression model:", result_df.iloc[0].to_dict())
    print("Top modularity snapshot:", modularity_df.sort_values('modularity_Q', ascending=False).iloc[0].to_dict())


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
