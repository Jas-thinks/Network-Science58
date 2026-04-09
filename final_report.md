# Final Report: Global Migration Network Deliverables

## 1. Project Summary
This project converts the UN DESA bilateral migration stock data into a dynamic directed network, community structure diagnostics, and regression-ready analytical outputs.

## 2. Key Deliverables Produced
- 8-snapshot dynamic DiGraph files in `deliverables_outputs/dynamic_graphs/`
- Centrality CSVs in `deliverables_outputs/centrality_csvs/`
- Community detection tables, similarity comparisons, and drift charts in `deliverables_outputs/community_detection/`
- Regression coefficient tables, R² comparison, and VIF report in `deliverables_outputs/regression_output/`
- Five dashboard screenshot exports and dashboard-ready CSVs in `deliverables_outputs/powerbi_exports/`

## 3. Dynamic Network Summary
- Snapshots analysed: 8
- Highest modularity snapshot: 1995 with Q = 0.5534
- Mean NMI across transitions: 0.7455
- Mean ARI across transitions: 0.6180
- Mean best-match Jaccard drift: 0.7214

## 4. Regression Summary
- Best model: random_forest
- Best test R²: 0.9455
- Best test RMSE: 0.6944
- Highest VIF feature: pair_prev_count (6.5386)

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
