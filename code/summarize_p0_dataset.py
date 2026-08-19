#!/usr/bin/env python3
"""Create the paper-facing multicity statistics table and split map."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CITY_LOCATIONS = {
    "Tokyo Chiyoda": (139.7536, 35.6940), "Tokyo Minato": (139.7514, 35.6581),
    "Tokyo Shinjuku": (139.7036, 35.6938), "Saitama": (139.6489, 35.8617),
    "Yokohama": (139.6380, 35.4437), "Kawasaki": (139.7029, 35.5308),
    "Osaka": (135.5023, 34.6937), "Fukuoka": (130.4017, 33.5904),
    "Hiroshima": (132.4553, 34.3853), "Sendai": (140.8719, 38.2682),
}
METRO_LABELS = {"Tokyo Chiyoda": "C", "Tokyo Minato": "M", "Tokyo Shinjuku": "S",
                "Saitama": "Sa", "Yokohama": "Y", "Kawasaki": "K"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--map", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    grouped = defaultdict(lambda: {"tiles": 0, "buildings": 0, "boundaries": 0})
    for item in manifest["models"]:
        row = grouped[(item["region"], item["split"])]
        row["tiles"] += 1
        row["buildings"] += int(item.get("building_count", 0))
        row["boundaries"] += int(item.get("boundary_count", 0))
    source_by_region = {item["region"]: item for item in manifest["sources"]}
    output = Path(args.csv); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "split", "year", "specification", "tiles", "buildings", "boundary_nodes"])
        for (region, split), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
            source = source_by_region[region]
            writer.writerow([region, split, source["year"], source["specification"],
                             values["tiles"], values["buildings"], values["boundaries"]])

    import matplotlib.pyplot as plt
    colors = {"train": "#2474b5", "validation": "#f39c35", "test": "#c53a32"}
    markers = {"train": "o", "validation": "s", "test": "^"}
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for split in ("train", "validation", "test"):
        rows = [(region, values) for (region, current), values in grouped.items() if current == split]
        xs = [CITY_LOCATIONS[region][0] for region, _ in rows]
        ys = [CITY_LOCATIONS[region][1] for region, _ in rows]
        sizes = [55 + values["tiles"] * 5 for _, values in rows]
        ax.scatter(xs, ys, s=sizes, c=colors[split], marker=markers[split],
                   edgecolor="white", linewidth=0.8, label=split.title(), zorder=3)
        for region, _values in rows:
            x, y = CITY_LOCATIONS[region]
            label = METRO_LABELS.get(region, region)
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)",
           title="Project PLATEAU cross-city benchmark split")
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    ax.text(0.02, 0.57, "Tokyo metro labels:\nC  Chiyoda   M  Minato   S  Shinjuku\n"
            "Sa  Saitama   Y  Yokohama   K  Kawasaki", transform=ax.transAxes,
            fontsize=8, va="top", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#dddddd"})
    target = Path(args.map); target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
