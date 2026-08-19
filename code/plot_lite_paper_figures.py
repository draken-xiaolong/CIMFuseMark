#!/usr/bin/env python3
"""Paper figures following LiteGeoFuseMark's robustness/uniqueness visual logic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

ATTACKS = [
    ("object_delete", "Object deletion"), ("building_delete", "Building deletion"),
    ("surface_delete", "Surface deletion"), ("attribute_delete", "Attribute deletion"),
    ("coordinate_noise", "Coordinate noise"), ("quantization", "Quantization"),
    ("rotation_z", "Z-axis rotation"), ("sequential", "Sequential attack"),
    ("spatial_crop", "Spatial crop"), ("hierarchy_flatten", "Hierarchy flattening"),
    ("semantic_relabel", "Semantic relabeling"), ("building_add", "Building addition"),
]

BASELINES = [
    ("jiang18_citygml_radial_histogram", "Jiang18", "#6F6F6F", ":", "^"),
    ("lee21_spherical_skew", "Lee21-adapted", "#8B6DAA", "-.", "v"),
    ("wang19_multifeature_adapted", "Wang19-adapted", "#D88A28", ":", "D"),
    ("hu26_radial_fusion_adapted", "Hu26-adapted", "#3A9D78", "-.", "P"),
    ("nonlearned_relation_graph", "Nonlearned graph", "#7B5A45", "--", "x"),
]


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6.2, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
        "xtick.labelsize": 5.1, "ytick.labelsize": 5.4, "axes.linewidth": 0.65,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save_all(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def attack_map(report: dict) -> dict:
    return {name: {float(row["intensity"]): row for row in rows}
            for name, rows in report["curves"].items()}


def robustness(base: dict, baseline_report: dict, stem: Path) -> list[dict]:
    style(); base_map = attack_map(base)
    baseline_maps = {key: {name: {float(row["intensity"]): row for row in values}
                           for name, values in baseline_report["methods"][key]["curves"].items()}
                     for key, *_ in BASELINES}
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.05), sharey=True); axes = axes.ravel()
    rows = []
    for panel, (ax, (attack, title)) in enumerate(zip(axes, ATTACKS)):
        levels = sorted(base_map.get(attack, {}))
        x = np.arange(len(levels)); base_y = [base_map[attack][level]["similarity_mean"] for level in levels]
        ax.plot(x, base_y, color="#C83E4D", linestyle="-", marker="s", markersize=3,
                linewidth=1.55, label="CIMFuseMark-Lite", zorder=8)
        row_by_level = {level: {"attack": attack, "intensity": level, "cimfusemark_lite": value}
                        for level, value in zip(levels, base_y)}
        for key, label, color, linestyle, marker in BASELINES:
            available = baseline_maps[key].get(attack, {})
            baseline_y = [available[level]["positive_mean"] for level in levels]
            ax.plot(x, baseline_y, color=color, linestyle=linestyle, marker=marker,
                    markersize=2.3, linewidth=.9, alpha=.9, label=label)
            for level, value in zip(levels, baseline_y): row_by_level[level][key] = value
        labels = [f"{level:g}" for level in levels]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35 if len(labels) > 5 else 0,
                                             ha="right" if len(labels) > 5 else "center")
        ax.set_ylim(-0.02, 1.02); ax.set_yticks(np.linspace(0, 1, 6)); ax.grid(axis="y", color="#E8E8E8", linewidth=.45)
        ax.set_title(title, pad=3, fontweight="semibold")
        ax.text(-0.15, 1.04, chr(97+panel), transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", va="bottom")
        if panel % 3 == 0: ax.set_ylabel("Mean NC")
        rows.extend(row_by_level[level] for level in levels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3,
               handlelength=2.4, columnspacing=1.2)
    fig.suptitle("CIMFuseMark-Lite robustness under CityGML attacks", y=1.01,
                 fontsize=9.2, fontweight="bold")
    fig.text(.5, .016, "All methods use the same 64 CityGML tiles and attacked XML files; mesh methods are explicit CityGML adaptations.",
             ha="center", fontsize=5.6, color="#555555")
    fig.subplots_adjust(left=.08, right=.985, bottom=.07, top=.84, wspace=.18, hspace=.54)
    save_all(fig, stem)
    return rows


def fingerprints(report: dict, ids: list[str]) -> np.ndarray:
    values = report["protocol"]["clean_fingerprints"]
    return np.asarray([[character == "1" for character in values[model_id]] for model_id in ids], dtype=np.uint8)


def similarity_matrix(bits: np.ndarray) -> np.ndarray:
    return (bits[:, None, :] == bits[None, :, :]).mean(axis=2)


def matrix_stats(matrix: np.ndarray) -> dict:
    values = matrix[~np.eye(len(matrix), dtype=bool)]
    return {"mean": float(values.mean()), "q95": float(np.quantile(values, .95)), "max": float(values.max())}


def draw_matrix(ax, matrix, title, panel, cmap, norm, show_y, boundaries, labels, outline=False):
    shown = matrix.copy(); np.fill_diagonal(shown, np.nan)
    image = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ticks = [7.5, 23.5, 39.5, 55.5]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(labels if show_y else [])
    ax.tick_params(length=0, pad=1.5); ax.set_title(title, pad=13, fontweight="semibold")
    ax.text(-.08, 1.09, panel, transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    for boundary in boundaries:
        ax.axhline(boundary, color="#252525", linewidth=.65); ax.axvline(boundary, color="#252525", linewidth=.65)
    if outline:
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i != j and matrix[i, j] >= .75:
                    ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="#111111", linewidth=.25))
    return image


def uniqueness(base_curves: dict, personal_curves: dict, manifest: dict, stem: Path) -> list[dict]:
    style(); items = sorted((item for item in manifest["models"] if item.get("split") == "test"),
                            key=lambda item: (item["region"], item["id"]))
    ids = [item["id"] for item in items]; regions = [item["region"] for item in items]
    base = similarity_matrix(fingerprints(base_curves, ids)); personal = similarity_matrix(fingerprints(personal_curves, ids))
    delta = personal-base; base_stats, personal_stats = matrix_stats(base), matrix_stats(personal)
    labels = list(dict.fromkeys(regions)); boundaries = []
    for index in range(len(regions)-1):
        if regions[index] != regions[index+1]: boundaries.append(index+.5)
    sequential = LinearSegmentedColormap.from_list("cim_nc", ["#F7FBFF", "#D9E8F1", "#91BED0", "#3D7FA3", "#173F66"])
    sequential.set_bad("#F2F2F2")
    diverging = LinearSegmentedColormap.from_list("cim_delta", ["#2867A0", "#A9CBE0", "#F7F7F7", "#E8AAA3", "#AF3E3C"])
    diverging.set_bad("#F2F2F2")
    fig = plt.figure(figsize=(10.3, 4.05)); grid = fig.add_gridspec(1, 5, width_ratios=[1,1,.045,1,.045], left=.055, right=.985, bottom=.14, top=.85, wspace=.14)
    axes = [fig.add_subplot(grid[0,0]), fig.add_subplot(grid[0,1]), fig.add_subplot(grid[0,3])]
    cax1, cax2 = fig.add_subplot(grid[0,2]), fig.add_subplot(grid[0,4])
    nc_norm = mpl.colors.Normalize(vmin=.35, vmax=.90); delta_norm = TwoSlopeNorm(vmin=-.40, vcenter=0, vmax=.15)
    im1 = draw_matrix(axes[0], base, f"Lite Base  mean={base_stats['mean']:.2f}, q95={base_stats['q95']:.2f}, max={base_stats['max']:.2f}", "a", sequential, nc_norm, True, boundaries, labels, True)
    draw_matrix(axes[1], personal, f"Personalized projection  mean={personal_stats['mean']:.2f}, q95={personal_stats['q95']:.2f}, max={personal_stats['max']:.2f}", "b", sequential, nc_norm, False, boundaries, labels, True)
    im2 = draw_matrix(axes[2], delta, "Personalized − Base pairwise change", "c", diverging, delta_norm, False, boundaries, labels, False)
    cb1 = fig.colorbar(im1, cax=cax1); cb1.set_label("Pairwise Hamming similarity", fontsize=6); cb1.ax.tick_params(labelsize=5)
    cb2 = fig.colorbar(im2, cax=cax2); cb2.set_label("Δ similarity (blue = improved)", fontsize=6); cb2.ax.tick_params(labelsize=5)
    off = ~np.eye(len(base), dtype=bool); changes = delta[off]
    fig.suptitle("CIMFuseMark-Lite cross-city zero-watermark uniqueness", y=.99, fontsize=9.5, fontweight="bold")
    fig.text(.5, .025, f"64 registered CIMs, 2,016 unique pairs; outlined cells ≥ 0.75. q95 {base_stats['q95']:.2f} → {personal_stats['q95']:.2f}; {100*np.mean(changes<0):.1f}% of pairs decrease.", ha="center", fontsize=5.7, color="#4A4A4A")
    save_all(fig, stem)
    return [{"left": ids[i], "right": ids[j], "base": float(base[i,j]), "personalized": float(personal[i,j]), "delta": float(delta[i,j])}
            for i in range(len(ids)) for j in range(i+1, len(ids))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-curves", required=True); parser.add_argument("--personal-curves", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    robust_rows = robustness(load(Path(args.base_curves)), load(Path(args.baselines)), output/"cimfusemark_lite_robustness_grid")
    unique_rows = uniqueness(load(Path(args.base_curves)), load(Path(args.personal_curves)), load(Path(args.manifest)), output/"cimfusemark_lite_uniqueness_matrix")
    for name, rows in (("cimfusemark_lite_robustness.csv", robust_rows), ("cimfusemark_lite_uniqueness.csv", unique_rows)):
        with (output/name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
