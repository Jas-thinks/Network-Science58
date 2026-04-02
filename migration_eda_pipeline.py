from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)


def read_table(path: Path, sheet: str | int | None = None, header: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, header=header)
    return pd.read_csv(path)


def normalize_year(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    return df


def standardize_country_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def melt_matrix_to_long(
    df: pd.DataFrame,
    origin_col: str,
    year: int | None = None,
) -> pd.DataFrame:
    """Convert a migration matrix (origin rows, destination columns) to long format.

    The resulting columns are: origin_country, destination_country, year, migrants.
    """
    if origin_col not in df.columns:
        raise ValueError(f"origin_col '{origin_col}' not in columns")

    value_cols = [c for c in df.columns if c != origin_col]
    long_df = df.melt(
        id_vars=[origin_col],
        value_vars=value_cols,
        var_name="destination_country",
        value_name="migrants",
    )
    long_df = long_df.rename(columns={origin_col: "origin_country"})
    if year is not None:
        long_df["year"] = int(year)
    return long_df


def load_un_desa_table1(path: Path) -> pd.DataFrame:
    """Load UN DESA destination-origin table (Table 1) into long format.

    This is already long (destination/origin with yearly columns). We melt years.
    """
    raw = pd.read_excel(path, sheet_name="Table 1", header=10)
    raw = raw.rename(columns={
        "Region, development group, country or area of destination": "destination_country",
        "Region, development group, country or area of origin": "origin_country",
        "Location code of destination": "dest_code",
        "Location code of origin": "orig_code",
    })
    years = [c for c in raw.columns if isinstance(c, (int, np.integer))]
    keep = ["destination_country", "origin_country", "dest_code", "orig_code", "Data type"] + years
    raw = raw[keep].copy()
    raw = raw.dropna(subset=["destination_country", "origin_country"], how="any")
    for y in years:
        raw[y] = pd.to_numeric(raw[y], errors="coerce")

    long_df = raw.melt(
        id_vars=["destination_country", "origin_country", "dest_code", "orig_code", "Data type"],
        value_vars=years,
        var_name="year",
        value_name="migrants",
    )
    return long_df


def remove_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop aggregate regions and income groups if Data type is missing.

    Works for UN DESA dataset. If Data type is absent, returns unchanged.
    """
    if "Data type" not in df.columns:
        return df
    aggregate_labels = set(
        df.loc[df["Data type"].isna(), "destination_country"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    mask = df["Data type"].notna()
    mask &= ~df["destination_country"].astype(str).str.strip().isin(aggregate_labels)
    mask &= ~df["origin_country"].astype(str).str.strip().isin(aggregate_labels)
    return df.loc[mask].copy()


def clean_migration_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = standardize_country_column(df, "origin_country")
    df = standardize_country_column(df, "destination_country")
    df = normalize_year(df, "year")

    df["migrants"] = pd.to_numeric(df["migrants"], errors="coerce")

    df = df[df["origin_country"] != df["destination_country"]]
    df = df.dropna(subset=["migrants", "origin_country", "destination_country", "year"])
    df = df[df["migrants"] > 0]
    return df


def standardize_auxiliary(df: pd.DataFrame, country_col: str, year_col: str, value_col: str) -> pd.DataFrame:
    out = df[[country_col, year_col, value_col]].copy()
    out = out.rename(columns={country_col: "country", year_col: "year", value_col: "value"})
    out = standardize_country_column(out, "country")
    out = normalize_year(out, "year")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def merge_origin_destination(
    df: pd.DataFrame,
    aux: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    aux_o = aux.rename(columns={"country": "origin_country", "value": f"{prefix}_origin"})
    aux_d = aux.rename(columns={"country": "destination_country", "value": f"{prefix}_destination"})

    merged = df.merge(aux_o, on=["origin_country", "year"], how="left")
    merged = merged.merge(aux_d, on=["destination_country", "year"], how="left")
    return merged


def handle_missing(df: pd.DataFrame, max_missing_ratio: float = 0.4) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_ratio = df[numeric_cols].isna().mean(axis=1)
    df = df[missing_ratio <= max_missing_ratio].copy()

    # Optional interpolation by year within origin/destination
    for col in numeric_cols:
        df[col] = df.groupby("origin_country")[col].transform(lambda s: s.interpolate())
        df[col] = df.groupby("destination_country")[col].transform(lambda s: s.interpolate())
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["gdp_origin", "gdp_destination", "population_origin", "population_destination"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    if "population_origin" in df.columns:
        df["migration_rate"] = df["migrants"] / df["population_origin"].replace(0, np.nan)
    return df


def basic_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    results["summary"] = df.describe(include="all")

    results["top_corridors"] = (
        df.groupby(["origin_country", "destination_country"], as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )

    results["top_destinations"] = (
        df.groupby("destination_country", as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )

    results["top_origins"] = (
        df.groupby("origin_country", as_index=False)["migrants"]
        .sum()
        .sort_values("migrants", ascending=False)
        .head(10)
    )

    return results


def correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="viridis", linewidths=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def prepare_ml(df: pd.DataFrame, target: str = "migrants") -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = {"origin_country", "destination_country"}
    if "dest_code" in df.columns:
        drop_cols.add("dest_code")
    if "orig_code" in df.columns:
        drop_cols.add("orig_code")
    if "Data type" in df.columns:
        drop_cols.add("Data type")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std(ddof=0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Migration EDA + ML prep pipeline")
    parser.add_argument("--migration", type=str, default=str(BASE / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"))
    parser.add_argument("--format", type=str, choices=["un_desa", "matrix"], default="un_desa")
    parser.add_argument("--origin-col", type=str, default="origin")
    parser.add_argument("--year", type=int, default=None)

    parser.add_argument("--gdp", type=str, default=None)
    parser.add_argument("--population", type=str, default=None)
    parser.add_argument("--unemployment", type=str, default=None)
    parser.add_argument("--education", type=str, default=None)

    parser.add_argument("--country-col", type=str, default="country")
    parser.add_argument("--year-col", type=str, default="year")
    parser.add_argument("--value-col", type=str, default="value")
    args = parser.parse_args()

    migration_path = Path(args.migration)
    if args.format == "un_desa":
        long_df = load_un_desa_table1(migration_path)
        long_df = remove_aggregates(long_df)
    else:
        matrix_df = read_table(migration_path)
        long_df = melt_matrix_to_long(matrix_df, origin_col=args.origin_col, year=args.year)

    long_df = clean_migration_long(long_df)

    merged = long_df.copy()

    aux_files = {
        "gdp": args.gdp,
        "population": args.population,
        "unemployment": args.unemployment,
        "education": args.education,
    }

    for name, path in aux_files.items():
        if path:
            aux = read_table(Path(path))
            aux = standardize_auxiliary(aux, args.country_col, args.year_col, args.value_col)
            merged = merge_origin_destination(merged, aux, name)

    merged = handle_missing(merged)
    merged = add_features(merged)

    merged.to_csv(OUT / "migration_cleaned.csv", index=False)

    eda = basic_eda(merged)
    for key, frame in eda.items():
        frame.to_csv(OUT / f"{key}.csv", index=False)

    correlation_heatmap(merged, OUT / "correlation_heatmap.png")

    X, y = prepare_ml(merged)
    X.to_csv(OUT / "ml_features.csv", index=False)
    y.to_csv(OUT / "ml_target.csv", index=False)

    print("Cleaned dataset saved to:", OUT / "migration_cleaned.csv")
    print("EDA outputs saved to:", OUT)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
