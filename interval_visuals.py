from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_clean" / "final_migration_dataset.csv"
VIS = BASE / "deliverables_outputs" / "visuals"
VIS.mkdir(parents=True, exist_ok=True)

YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["year"].isin(YEARS)].copy()
    return df


def total_migrants_by_year(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("year", as_index=False)["migrants"].sum()


def interval_changes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end in zip(YEARS[:-1], YEARS[1:]):
        total_start = df[df["year"] == start]["migrants"].sum()
        total_end = df[df["year"] == end]["migrants"].sum()
        rows.append({
            "interval": f"{start}-{end}",
            "start": total_start,
            "end": total_end,
            "delta": total_end - total_start,
            "pct_change": (total_end - total_start) / total_start * 100 if total_start else np.nan,
        })
    return pd.DataFrame(rows)


def plot_interval_totals(df_year: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(df_year["year"], df_year["migrants"], marker="o")
    plt.title("Total Migration Stock by Snapshot (1990–2024)")
    plt.xlabel("Year")
    plt.ylabel("Migrants")
    plt.tight_layout()
    plt.savefig(VIS / "interval_total_trend.png", dpi=200)
    plt.close()


def plot_interval_changes(interval_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in interval_df["delta"]]
    plt.bar(interval_df["interval"], interval_df["delta"], color=colors)
    plt.title("Change in Migration Stock by 5-Year Interval")
    plt.xlabel("Interval")
    plt.ylabel("Delta Migrants")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(VIS / "interval_delta_bar.png", dpi=200)
    plt.close()


def plot_top_corridors_by_interval(df: pd.DataFrame) -> None:
    for start, end in zip(YEARS[:-1], YEARS[1:]):
        end_df = df[df["year"] == end]
        top = (
            end_df.groupby(["origin", "destination"], as_index=False)["migrants"]
            .sum()
            .sort_values("migrants", ascending=False)
            .head(10)
        )
        plt.figure(figsize=(10, 5))
        labels = top["origin"] + " → " + top["destination"]
        plt.barh(labels, top["migrants"], color="#1f77b4")
        plt.gca().invert_yaxis()
        plt.title(f"Top Corridors at {end} (Interval {start}-{end})")
        plt.xlabel("Migrants")
        plt.tight_layout()
        plt.savefig(VIS / f"interval_top_corridors_{start}_{end}.png", dpi=200)
        plt.close()


def create_collage() -> None:
    # Define a collage grid from existing visuals
    paths = [
        VIS / "network_pagerank_community.png",
        VIS / "map_degree.png",
        VIS / "map_betweenness.png",
        VIS / "map_closeness.png",
        VIS / "map_pagerank.png",
        VIS / "migration_corridors_map.png",
        VIS / "interval_total_trend.png",
        VIS / "interval_delta_bar.png",
    ]
    images = [Image.open(p).convert("RGB") for p in paths if p.exists()]
    if not images:
        return

    # Resize to uniform thumbnails
    thumb_w, thumb_h = 800, 450
    thumbs = [img.resize((thumb_w, thumb_h)) for img in images]

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    collage = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))

    for idx, img in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        collage.paste(img, (c * thumb_w, r * thumb_h))

    collage.save(VIS / "final_collage.png")


def main() -> None:
    df = load_data()
    totals = total_migrants_by_year(df)
    interval_df = interval_changes(df)

    totals.to_csv(VIS / "interval_totals.csv", index=False)
    interval_df.to_csv(VIS / "interval_changes.csv", index=False)

    plot_interval_totals(totals)
    plot_interval_changes(interval_df)
    plot_top_corridors_by_interval(df)
    create_collage()

    print(f"Interval visuals saved to {VIS}")


if __name__ == "__main__":
    main()
