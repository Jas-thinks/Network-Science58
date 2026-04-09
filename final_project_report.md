# Final Project Report

**Title:** Global Migration Network Analysis: Data Science and Network Science Approaches

**Authors:**
- Jaswinder Singh (ID: 12340970)
- Aditya Yadav
- Dipanjan Mondal

---

## 1. Project Aim
The aim of this project was to analyze global migration flows using advanced data science and network science techniques. We set out to:
- Clean and integrate international migration datasets
- Construct dynamic migration networks
- Detect communities and analyze their evolution
- Model migration determinants using regression
- Build interactive dashboards for exploration

## 2. Data Collection
- **Primary Source:** United Nations Department of Economic and Social Affairs (UN DESA) International Migrant Stock 2024 Excel file (undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx)
- **Auxiliary Data:** World Bank (GDP, population, unemployment), UNDP, UCDP (conflict), and other open sources (where available)
- **Collaboration:** Integrated and compared with the open-source Nodes_And_Nations repository for advanced pipelines and validation

## 3. Data Preparation & Cleaning
- Parsed the UN DESA Excel into long-format bilateral migration flows (country-to-country, by year)
- Cleaned and filtered out non-country and aggregate rows
- Merged auxiliary factor data (where available) on (country, year)
- Handled missing values and harmonized country codes
- Exported cleaned datasets for analysis

## 4. Network Construction & Centrality
- Built 8 dynamic directed migration networks (snapshots: 1990–2024/2025)
- Nodes: countries; Edges: migration flows (weighted)
- Computed centrality metrics: in-degree, out-degree, betweenness, PageRank
- Exported centrality CSVs for each snapshot

## 5. Community Detection & Temporal Analysis
- Applied Louvain, Leiden, and Girvan-Newman algorithms to detect communities in each network snapshot
- Calculated modularity Q for each method and year
- Compared partitions using Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), and Jaccard drift
- Analyzed community stability and boundary nodes over time

## 6. Regression Analysis
- Merged centrality and factor data for regression modeling
- Fitted baseline and full OLS models (log-migrants ~ log-population, log-GDP, conflict, etc.)
- Compared models using R², AIC, and VIF (Variance Inflation Factor)
- Exported coefficient tables and diagnostics

## 7. Dashboarding & Visualization
- Generated dashboard-ready CSVs and Power BI screenshot exports for:
  - Executive overview
  - Top migration corridors
  - Temporal trends
  - Community structure
  - Regression diagnostics
- Provided a Power BI build spec for interactive dashboard recreation

## 8. Algorithms Used
- **Network Construction:** NetworkX DiGraph, edge aggregation
- **Centrality:** in-degree, out-degree, betweenness, PageRank (NetworkX)
- **Community Detection:**
  - Louvain (python-louvain)
  - Leiden (leidenalg, igraph)
  - Girvan-Newman (NetworkX)
- **Partition Comparison:** NMI, ARI (scikit-learn), Jaccard drift
- **Regression:** OLS (statsmodels), VIF analysis
- **Visualization:** matplotlib, seaborn, Power BI

## 9. Workflow Summary
1. Data collection from UN DESA and APIs
2. Data cleaning and integration
3. Network construction for each snapshot
4. Centrality and community analysis
5. Regression modeling
6. Dashboard and report generation

## 10. Collaboration & Validation
- Collaborated with the Nodes_And_Nations open-source project for pipeline validation and advanced analytics
- Compared outputs and algorithms for robustness

## 11. Key Results
- Identified top migration corridors and hubs
- Detected persistent and shifting communities
- Regression models explained migration determinants with high accuracy (best R² ≈ 0.95)
- Produced reproducible, dashboard-ready outputs

## 12. References
- United Nations, Department of Economic and Social Affairs. *International Migrant Stock 2024*.
- World Bank Open Data
- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks.
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities.

---

**This report documents the full workflow, algorithms, and results for the global migration network analysis project by Jaswinder Singh, Aditya Yadav, and Dipanjan Mondal.**
