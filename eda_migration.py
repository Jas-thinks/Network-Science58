from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)


MAIN_FILE = BASE / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"
SHEETS = ["Table 1", "Table 2", "Table 9"]
YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]


def load_table1(path: Path) -> pd.DataFrame:
    """Load the main bilateral migration stock table.

    The workbook has metadata rows at the top; row 11 (header=10) contains
    the actual field names for the combined-sexes section.
    """
    df = pd.read_excel(path, sheet_name="Table 1", header=10)
    df = df.rename(columns={
        "Region, development group, country or area of destination": "destination",
        "Region, development group, country or area of origin": "origin",
        "Location code of destination": "dest_code",
        "Location code of origin": "orig_code",
    })
    return df


def clean_main_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["destination", "dest_code", "origin", "orig_code", "Data type"] + YEARS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df[keep].copy()
    out = out.dropna(subset=["destination", "origin"], how="any")
    out["destination"] = out["destination"].astype(str).str.strip()
    out["origin"] = out["origin"].astype(str).str.strip()
    out["dest_code"] = pd.to_numeric(out["dest_code"], errors="coerce")
    out["orig_code"] = pd.to_numeric(out["orig_code"], errors="coerce")
    for y in YEARS:
        out[y] = pd.to_numeric(out[y], errors="coerce")
    return out


def country_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep country/territory rows and drop aggregate regions.

    In this workbook, aggregate regions and income groups usually have a missing
    "Data type" field, while country/territory rows have labels such as B, C,
    B R, C R, I, or I R.
    """
    if "Data type" not in df.columns:
        return df.copy()
    aggregate_labels = set(
        df.loc[df["Data type"].isna(), "destination"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    mask = df["Data type"].notna()
    mask &= ~df["destination"].astype(str).str.strip().isin(aggregate_labels)
    mask &= ~df["origin"].astype(str).str.strip().isin(aggregate_labels)
    return df[mask].copy()


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["destination", "dest_code", "origin", "orig_code"],
        value_vars=YEARS,
        var_name="year",
        value_name="migrant_stock",
    )
    long_df = long_df.dropna(subset=["migrant_stock"])
    long_df = long_df[long_df["migrant_stock"] > 0].copy()
    long_df["year"] = long_df["year"].astype(int)
    return long_df


def basic_summary(df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "metric": [
            "rows_raw",
            "rows_clean",
            "unique_destinations",
            "unique_origins",
            "unique_pairs",
            "years",
            "nonzero_edges",
            "missing_destination_codes",
            "missing_origin_codes",
        ],
        "value": [
            len(df),
            len(df.dropna(subset=["destination", "origin"])),
            df["destination"].nunique(),
            df["origin"].nunique(),
            df[["destination", "origin"]].drop_duplicates().shape[0],
            len(YEARS),
            len(long_df),
            int(df["dest_code"].isna().sum()),
            int(df["orig_code"].isna().sum()),
        ]
    })


def top_totals(long_df: pd.DataFrame, n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    dest = (
        long_df.groupby("destination", as_index=False)["migrant_stock"]
        .sum()
        .sort_values("migrant_stock", ascending=False)
        .head(n)
    )
    origin = (
        long_df.groupby("origin", as_index=False)["migrant_stock"]
        .sum()
        .sort_values("migrant_stock", ascending=False)
        .head(n)
    )
    return dest, origin


def year_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    return (
        long_df.groupby("year", as_index=False)["migrant_stock"]
        .agg(count="count", total="sum", mean="mean", median="median")
    )


def duplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["destination", "origin"], as_index=False)
        .size()
        .query("size > 1")
        .sort_values("size", ascending=False)
    )


def main() -> None:
    df = load_table1(MAIN_FILE)
    clean = clean_main_table(df)
    country = country_only(clean)
    long_df = to_long(clean)
    country_long = to_long(country)

    summary = basic_summary(clean, long_df)
    summary.to_csv(OUT / "summary.csv", index=False)

    year_df = year_summary(long_df)
    year_df.to_csv(OUT / "year_summary.csv", index=False)

    country_summary = basic_summary(country, country_long)
    country_summary.to_csv(OUT / "country_summary.csv", index=False)

    country_year_df = year_summary(country_long)
    country_year_df.to_csv(OUT / "country_year_summary.csv", index=False)

    dest_top, origin_top = top_totals(long_df)
    dest_top.to_csv(OUT / "top_destinations.csv", index=False)
    origin_top.to_csv(OUT / "top_origins.csv", index=False)

    c_dest_top, c_origin_top = top_totals(country_long)
    c_dest_top.to_csv(OUT / "country_top_destinations.csv", index=False)
    c_origin_top.to_csv(OUT / "country_top_origins.csv", index=False)

    dup = duplicate_pairs(clean)
    dup.to_csv(OUT / "duplicate_pairs.csv", index=False)

    print("=== BASIC SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== COUNTRY-ONLY SUMMARY ===")
    print(country_summary.to_string(index=False))
    print("\n=== YEAR SUMMARY ===")
    print(year_df.to_string(index=False))
    print("\n=== COUNTRY-ONLY YEAR SUMMARY ===")
    print(country_year_df.to_string(index=False))
    print("\n=== TOP DESTINATIONS ===")
    print(dest_top.to_string(index=False))
    print("\n=== TOP ORIGINS ===")
    print(origin_top.to_string(index=False))
    print("\n=== COUNTRY-ONLY TOP DESTINATIONS ===")
    print(c_dest_top.to_string(index=False))
    print("\n=== COUNTRY-ONLY TOP ORIGINS ===")
    print(c_origin_top.to_string(index=False))
    print(f"\nOutputs saved to: {OUT}")


if __name__ == "__main__":
    main()
