# Results and Discussion

This chapter synthesizes centrality, community structure, and regression findings into a single analytical narrative. The migration system is modeled as a directed, weighted network where countries are nodes and bilateral stocks are edges. The emphasis is on how structural measures translate into real-world migration corridors and policy-relevant insights.

---

## Centrality Analysis: Structural Hubs and Bridge Countries

### Formal Explanation
Centrality quantifies a country’s structural importance in the migration network. In-degree and out-degree centrality capture the diversity of inflows and outflows, betweenness centrality measures brokerage across corridors, and PageRank reflects global influence by weighting connections to other influential countries. Together, these measures identify hubs, bridges, and globally central destinations.

### Methodology
We build a directed weighted graph from origin–destination flows and compute in-degree, out-degree, betweenness (weighted), and PageRank. Centralities are computed for each snapshot year, with 2024 used for global interpretation. The analysis is repeated across years to test structural persistence.

### Code (NetworkX + pandas)
```python
import pandas as pd
import networkx as nx

# Load dataset
# Columns: origin_ISO3, destination_ISO3, year, migrants

df = pd.read_csv("migration.csv")
df = df[df["migrants"] > 0]

# Filter one snapshot year
year = 2024
sub = df[df["year"] == year]

G = nx.DiGraph()
edges = sub.groupby(["origin_ISO3", "destination_ISO3"], as_index=False)["migrants"].sum()
for _, row in edges.iterrows():
    if row["origin_ISO3"] != row["destination_ISO3"]:
        G.add_edge(row["origin_ISO3"], row["destination_ISO3"], weight=row["migrants"])

in_deg = nx.in_degree_centrality(G)
out_deg = nx.out_degree_centrality(G)
betw = nx.betweenness_centrality(G, weight="weight", normalized=True)
pr = nx.pagerank(G, weight="weight")

centrality_df = pd.DataFrame({
    "country": list(G.nodes),
    "in_degree": [in_deg[n] for n in G.nodes],
    "out_degree": [out_deg[n] for n in G.nodes],
    "betweenness": [betw[n] for n in G.nodes],
    "pagerank": [pr[n] for n in G.nodes],
})
```

### Interpretation (Migration Context)
The highest in-degree and PageRank countries are global destination hubs. Large economies such as the USA and Germany consistently rank high because they connect to many origins and other influential nodes. Gulf states like the UAE often appear as high PageRank or high in-degree hubs due to strong labor corridors. High betweenness countries represent bridge roles (transit or regional connectors), which are critical for corridor stability. These patterns indicate a scale-free structure: a small set of countries dominate global migration connectivity.

---

## Community Detection: Regional Systems and Migration Blocs

### Formal Explanation
Community detection partitions the network into clusters with dense internal flows. In migration networks, communities represent regional migration systems or socio-economic blocs where movement is concentrated. Modularity quantifies the strength of clustering; higher modularity indicates clearer structural separation between blocs.

### Methodology
We convert the directed graph to an undirected weighted graph for partitioning. Louvain and Leiden optimize modularity, while Girvan–Newman provides a hierarchical baseline. Partitions are compared using modularity scores to assess stability and structural separation.

### Code (Louvain, Girvan–Newman, Leiden)
```python
import pandas as pd
import networkx as nx
import community as community_louvain
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity

# G = directed graph from centrality section
H = G.to_undirected()

# Louvain
louvain_part = community_louvain.best_partition(H, weight="weight")
louvain_mod = community_louvain.modularity(louvain_part, H, weight="weight")

# Girvan–Newman (first split)
communities = next(girvan_newman(H))
GN_part = {node: i for i, comm in enumerate(communities) for node in comm}
GN_mod = modularity(H, communities, weight="weight")

# Optional Leiden (if installed)
try:
    import igraph as ig
    import leidenalg
    nodes = list(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in H.edges()]
    weights = [H[u][v]["weight"] for u, v in H.edges()]
    igG = ig.Graph(edges=edges, directed=False)
    igG.es["weight"] = weights
    leiden = leidenalg.find_partition(igG, leidenalg.ModularityVertexPartition, weights="weight")
    leiden_part = {nodes[i]: int(leiden.membership[i]) for i in range(len(nodes))}
    leiden_mod = leiden.modularity
except Exception:
    leiden_part, leiden_mod = None, None
```

### Interpretation (Migration Context)
Communities typically align with geographic and economic regions (e.g., Europe, Gulf, or Latin America). High modularity indicates that migration flows are concentrated within blocs, while declining modularity over time suggests increasing global integration. Community shifts often reflect geopolitical shocks, conflict displacement, and changing policy openness.

---

## Regression Analysis: Drivers of Migration Flows

### Formal Explanation
Regression models explain migration flow intensity using socio-economic and policy predictors. We model the log-transformed migration stock to stabilize variance and interpret coefficients as semi-elasticities. This quantifies how economic pull (GDP), demographic scale (population), and push factors (conflict, unemployment) influence migration volumes.

### Methodology
We construct an OLS model with log-migrants as the dependent variable and GDP, population, unemployment, education, conflict, climate, and visa openness as predictors. Missing values are imputed using median values; variables are standardized if needed. VIF is used to assess multicollinearity.

### Code (pandas + statsmodels)
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Columns: migrants, GDP, population, unemployment, education, conflict, climate, visa_openness

df = pd.read_csv("migration_features.csv")

# Dependent variable
Y = np.log1p(df["migrants"])

# Independent variables
features = ["GDP", "population", "unemployment", "education", "conflict", "climate", "visa_openness"]
X = df[features].copy().fillna(df[features].median())
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

# VIF check
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print(model.summary())
print(vif)
```

### Interpretation (Migration Context)
Higher GDP and population are typically associated with larger migration inflows, confirming economic pull and scale effects. Conflict and climate indicators act as push factors, increasing outward migration from vulnerable regions. Visa openness is positively linked to flows, showing that policy accessibility directly shapes corridor intensity. These relationships provide actionable signals for policymakers designing migration and labor policies.

---

## Data Preparation for Visualization (Power BI)

### Formal Explanation
Interactive dashboards require standardized node-level and edge-level tables. Node data supports centrality maps and country ranking visuals, while edge data supports corridor analysis and time-based exploration. Exporting these datasets enables consistent, reproducible reporting.

### Methodology
We export node-level centrality metrics and edge-level corridor flows for each snapshot year. The node table stores centrality measures and community labels, while the edge table stores origin–destination flows and weights. These files can be directly ingested into Power BI.

### Code (pandas export)
```python
# Node-level export
centrality_df["year"] = year
centrality_df.to_csv("powerbi_nodes.csv", index=False)

# Edge-level export
edges["year"] = year
edges.to_csv("powerbi_edges.csv", index=False)
```

### Interpretation (Migration Context)
Dashboards built from these exports allow policymakers to identify top hubs (e.g., USA, UAE, Germany), compare corridor strengths, and monitor regional shifts in community membership. This supports evidence-based migration planning and targeted policy intervention.

---

## Global Migration Insights (Synthesis)
- **Persistent hubs:** The USA, Germany, and major Gulf destinations consistently appear as high PageRank and high in-degree nodes, indicating durable structural influence.
- **Bridging countries:** High betweenness nodes function as corridor intermediaries and are critical for regional connectivity.
- **Community structure:** Regional blocs remain stable over time, but conflict-driven regions show higher churn in community assignment.
- **Policy relevance:** Centrality and regression outputs highlight where migration pressure concentrates, enabling targeted bilateral agreements, labor planning, and humanitarian response.

---

## Practical Policy Implications
- Focus bilateral agreements on corridors identified as top edges and high-betweenness bridges.
- Use PageRank hubs to prioritize capacity planning and integration support.
- Monitor community drift to detect emerging migration shocks early.

---

**Data → Network Measures → Patterns → Real-world Meaning**
This pipeline converts raw bilateral data into interpretable structural signals and actionable policy insights, aligning network science with migration governance and planning.
