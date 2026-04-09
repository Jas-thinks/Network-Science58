from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import importlib.util

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "data_clean"
OUT_DIR.mkdir(exist_ok=True)


def read_table(path: Path, sheet: Optional[str] = None, header: Optional[int] = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, header=header)
    return pd.read_csv(path)


def load_un_desa_table1(path: Path) -> pd.DataFrame:
    """Load UN DESA destination-origin table (Table 1) into long format."""
    raw = pd.read_excel(path, sheet_name="Table 1", header=10)
    raw = raw.rename(columns={
        "Region, development group, country or area of destination": "destination",
        "Region, development group, country or area of origin": "origin",
        "Location code of destination": "dest_code",
        "Location code of origin": "orig_code",
    })
    years = [c for c in raw.columns if isinstance(c, (int, np.integer))]
    keep = ["destination", "origin", "dest_code", "orig_code", "Data type"] + years
    raw = raw[keep].copy()
    raw = raw.dropna(subset=["destination", "origin"], how="any")
    for y in years:
        raw[y] = pd.to_numeric(raw[y], errors="coerce")

    long_df = raw.melt(
        id_vars=["destination", "origin", "dest_code", "orig_code", "Data type"],
        value_vars=years,
        var_name="year",
        value_name="migrants",
    )
    return long_df


def remove_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop aggregate regions and income groups if Data type is missing."""
    if "Data type" not in df.columns:
        return df
    aggregate_labels = set(
        df.loc[df["Data type"].isna(), "destination"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    mask = df["Data type"].notna()
    mask &= ~df["destination"].astype(str).str.strip().isin(aggregate_labels)
    mask &= ~df["origin"].astype(str).str.strip().isin(aggregate_labels)
    return df.loc[mask].copy()


def to_iso3(series: pd.Series) -> pd.Series:
    """Convert country names to ISO3 if possible.

    If the input already looks like ISO3, it is kept as-is.
    Attempts to use pycountry when available; otherwise leaves values unchanged.
    """
    s = series.astype(str).str.strip()

    def _convert(val: str) -> str:
        if len(val) == 3 and val.isalpha():
            return val.upper()
        if importlib.util.find_spec("pycountry") is None:
            return val
        try:
            pycountry = importlib.import_module("pycountry")
            match = pycountry.countries.search_fuzzy(val)
            if match:
                return match[0].alpha_3
        except Exception:
            return val
        return val

    return s.apply(_convert)


def melt_migration_matrix(df: pd.DataFrame, origin_col: str, year: Optional[int] = None) -> pd.DataFrame:
    if origin_col not in df.columns:
        raise ValueError(f"origin_col '{origin_col}' not in migration matrix")

    value_cols = [c for c in df.columns if c != origin_col]
    long_df = df.melt(
        id_vars=[origin_col],
        value_vars=value_cols,
        var_name="destination",
        value_name="migrants",
    ).rename(columns={origin_col: "origin"})

    if year is not None:
        long_df["year"] = int(year)
    return long_df


def clean_migration(df: pd.DataFrame, origin_col: str = "origin", dest_col: str = "destination") -> pd.DataFrame:
    out = df.copy()
    out[origin_col] = out[origin_col].astype(str).str.strip()
    out[dest_col] = out[dest_col].astype(str).str.strip()
    out["migrants"] = pd.to_numeric(out["migrants"], errors="coerce")

    out = out[out[origin_col] != out[dest_col]]
    out = out.dropna(subset=["migrants", origin_col, dest_col, "year"])
    out = out[out["migrants"] > 0]

    out[origin_col] = to_iso3(out[origin_col])
    out[dest_col] = to_iso3(out[dest_col])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)

    return out


def clean_auxiliary(
    df: pd.DataFrame,
    country_col: str,
    year_col: str,
    value_col: str,
    fill_strategy: str = "interpolate",
) -> pd.DataFrame:
    out = df[[country_col, year_col, value_col]].copy()
    out = out.rename(columns={country_col: "country", year_col: "year", value_col: "value"})
    out["country"] = to_iso3(out["country"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if fill_strategy == "interpolate":
        out["value"] = out.groupby("country")["value"].transform(lambda s: s.interpolate())
    elif fill_strategy == "zero":
        out["value"] = out["value"].fillna(0)

    return out


def merge_origin_destination(base: pd.DataFrame, aux: pd.DataFrame, prefix: str) -> pd.DataFrame:
    aux_o = aux.rename(columns={"country": "origin", "value": f"{prefix}_origin"})
    aux_d = aux.rename(columns={"country": "destination", "value": f"{prefix}_destination"})

    merged = base.merge(aux_o, on=["origin", "year"], how="left")
    merged = merged.merge(aux_d, on=["destination", "year"], how="left")
    return merged


def merge_pair_features(base: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    pair_df = pair_df.rename(columns={
        "origin": "origin",
        "destination": "destination",
    })
    return base.merge(pair_df, on=["origin", "destination"], how="left")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "gdp_origin",
        "gdp_destination",
        "population_origin",
        "population_destination",
        "distance_km",
    ]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col])

    if "population_origin" in out.columns:
        out["migration_rate"] = out["migrants"] / out["population_origin"].replace(0, np.nan)

    if "gdp_destination" in out.columns and "gdp_origin" in out.columns:
        out["gdp_diff"] = out["gdp_destination"] - out["gdp_origin"]

    return out


def final_cleaning(df: pd.DataFrame, max_missing_ratio: float = 0.4) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["origin", "destination", "year"]).copy()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    missing_ratio = out[numeric_cols].isna().mean(axis=1)
    out = out[missing_ratio <= max_missing_ratio].copy()

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Global migration master dataset pipeline")

    parser.add_argument("--migration", required=True, help="Migration file path")
    parser.add_argument("--format", choices=["matrix", "un_desa"], default="matrix")
    parser.add_argument("--origin-col", default="origin", help="Origin column in migration matrix")
    parser.add_argument("--year", type=int, default=None, help="Year for migration matrix if single-year")

    parser.add_argument("--gdp", default=None)
    parser.add_argument("--population", default=None)
    parser.add_argument("--unemployment", default=None)
    parser.add_argument("--remittances", default=None)
    parser.add_argument("--stability", default=None)
    parser.add_argument("--internet", default=None)
    parser.add_argument("--education", default=None)
    parser.add_argument("--conflict", default=None)
    parser.add_argument("--climate", default=None)
    parser.add_argument("--visa", default=None)

    parser.add_argument("--pair", default=None, help="CEPII GeoDist pair features file")

    parser.add_argument("--country-col", default="country")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--value-col", default="value")

    args = parser.parse_args()

    migration_path = Path(args.migration)
    if args.format == "un_desa":
        long_df = load_un_desa_table1(migration_path)
        long_df = remove_aggregates(long_df)
    else:
        migration_df = read_table(migration_path)
        long_df = melt_migration_matrix(migration_df, origin_col=args.origin_col, year=args.year)
    long_df = clean_migration(long_df)

    merged = long_df.copy()

    aux_inputs = {
        "gdp": (args.gdp, "interpolate"),
        "population": (args.population, "interpolate"),
        "unemployment": (args.unemployment, "interpolate"),
        "remittances": (args.remittances, "interpolate"),
        "stability": (args.stability, "interpolate"),
        "internet": (args.internet, "interpolate"),
        "education": (args.education, "interpolate"),
        "conflict": (args.conflict, "zero"),
        "climate": (args.climate, "zero"),
        "visa": (args.visa, "interpolate"),
    }

    for name, (path, strategy) in aux_inputs.items():
        if path:
            aux_raw = read_table(Path(path))
            aux = clean_auxiliary(aux_raw, args.country_col, args.year_col, args.value_col, fill_strategy=strategy)
            merged = merge_origin_destination(merged, aux, name)

    if args.pair:
        pair_df = read_table(Path(args.pair))
        pair_df = pair_df.rename(columns={
            "origin": "origin",
            "destination": "destination",
        })
        merged = merge_pair_features(merged, pair_df)

    merged = feature_engineering(merged)
    merged = final_cleaning(merged)

    output_path = OUT_DIR / "final_migration_dataset.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
