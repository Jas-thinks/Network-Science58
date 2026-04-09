# Power BI Dashboard Build Spec

The following dashboard-ready CSVs were exported to `deliverables_outputs/powerbi_exports/data/`.

## Dashboard 1: Executive Overview
**Data:** `dashboard1_overview.csv`, `dashboard3_temporal.csv`, `dashboard4_communities.csv`, `dashboard5_regression.csv`
**Visuals:** KPI cards for total migrants, line chart for trend, bar chart for top destinations, modularity line, R² bar chart.

## Dashboard 2: Corridor Analysis
**Data:** `dashboard2_corridors.csv`
**Visuals:** Ranked corridor bar chart, flow concentration pie chart, country slicers for origin/destination.

## Dashboard 3: Temporal Dynamics
**Data:** `dashboard3_temporal.csv`, community similarity tables
**Visuals:** time-series line chart, NMI/ARI trend, Jaccard drift chart, snapshot slicer.

## Dashboard 4: Community Structure
**Data:** `dashboard4_communities.csv`, `modularity_Q_table.csv`, `nmi_ari_comparison.csv`
**Visuals:** modularity table, community count trend, snapshot matrix, highlighted nodes by community.

## Dashboard 5: Regression Diagnostics
**Data:** `dashboard5_regression.csv`, `coefficient_tables.csv`, `vif_report.csv`
**Visuals:** model comparison bar chart, coefficient table, VIF bar chart, residual diagnostics.

## Export Notes
- Screenshot exports are in `deliverables_outputs/powerbi_exports/screenshots/`.
- Native `.pbix` authoring is not available in this Linux workspace.
- To complete the `.pbix` files, import the CSVs into Power BI Desktop and recreate the visuals described above.
