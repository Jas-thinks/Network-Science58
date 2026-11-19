# Global Migration Network Analysis

Network-science and machine-learning analysis of international migration, modelling countries as nodes and migrant stocks as weighted, directed edges across eight snapshots (1990–2024).

## What it does

| Stage | Output |
|---|---|
| **EDA** (`eda_migration.py`) | Country, country-year and corridor summaries, duplicate-pair checks |
| **Dynamic networks** (`deliverables_pipeline.py`) | Directed weighted graph per snapshot year (1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024) |
| **Centrality** | Degree, betweenness, closeness and PageRank per country and year |
| **Community detection** | Communities per snapshot, modularity, NMI/ARI comparison, Jaccard drift (community stability over time) |
| **Regression** | Linear, Ridge and Lasso models with VIF multicollinearity checks and R² comparison |
| **Visualisation** (`centrality_visuals.py`) | Centrality histograms, corridor maps, interval trend charts, interactive HTML maps |
| **Dashboard** | Power BI dashboard screenshots (overview, corridors, temporal, communities, regression) |

## Repository layout

```text
.
├── data_clean/                 # Cleaned migration dataset (destination, origin, year, migrants)
├── eda_migration.py            # Exploratory data analysis
├── deliverables_pipeline.py    # Graph construction, centrality, communities, regression
├── centrality_visuals.py       # Charts and maps
├── eda_outputs/                # EDA tables
├── deliverables_outputs/       # Centrality CSVs, communities, regression, visuals, dashboard exports
└── *.pdf                       # Project reports
```

## Reproducing

```bash
pip install pandas numpy networkx scikit-learn statsmodels matplotlib geopandas folium
python eda_migration.py
python deliverables_pipeline.py
python centrality_visuals.py
```

Scripts may contain absolute paths from the original machine; adjust the data/output directories at the top of each script if needed.

## Reports

See the PDF reports in the repository root for methodology and findings.
