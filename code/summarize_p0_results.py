#!/usr/bin/env python3
"""Turn P0 raw JSON files into paper tables and diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("base", "separated", "personalized", "deepsets")
DISPLAY = {"base": "Base", "separated": "Separation", "personalized": "Personalized",
           "deepsets": "DeepSets/no-edge"}
SELECTED_ATTACKS = ("building_delete", "surface_delete", "lod2_to_lod1",
                    "hierarchy_flatten", "semantic_relabel", "cityjson_roundtrip")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rate_curve(positives: list[float], negatives: list[float]):
    thresholds = np.unique(np.asarray([0.0, 1.0, *positives, *negatives]))
    fars = np.asarray([np.mean(np.asarray(negatives) >= value) for value in thresholds])
    tars = np.asarray([np.mean(np.asarray(positives) >= value) for value in thresholds])
    order = np.argsort(fars)
    return fars[order], tars[order], thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); results = Path(args.results); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    curves, opens = {}, {}
    for variant in VARIANTS:
        curve_path = results / f"multicity_{variant}_curves.json"
        open_path = results / f"multicity_{variant}_open_set.json"
        if curve_path.exists(): curves[variant] = load(curve_path)
        if open_path.exists(): opens[variant] = load(open_path)
    if not curves:
        raise SystemExit("No completed multicity model curves found")

    with (output / "p0_model_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "models", "bits", "distinct", "collision_pairs",
                         "negative_mean", "negative_q95", "negative_max", "far5_threshold", "far1_threshold"])
        for variant, report in curves.items():
            p = report["protocol"]; opened = opens.get(variant, {}).get("open_set", {})
            writer.writerow([DISPLAY[variant], p["models"], p["fingerprint_bits"],
                             p.get("distinct_fingerprints"), p.get("collision_pairs"),
                             p["negative_mean"], p["negative_q95"], p["negative_maximum"],
                             opened.get("thresholds", {}).get("far_5pct"),
                             opened.get("thresholds", {}).get("far_1pct")])

    with (output / "p0_attack_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "attack", "intensity", "mean", "q05", "auc", "eer",
                         "semantic_ratio", "coordinate_ratio", "tar_at_far5", "tar_at_far1"])
        for variant, report in curves.items():
            fixed = {attack: {str(row["intensity"]): row for row in rows}
                     for attack, rows in opens.get(variant, {}).get("fixed_threshold_attacks", {}).items()}
            for attack, points in report["curves"].items():
                for point in points:
                    threshold = fixed.get(attack, {}).get(str(point["intensity"]), {})
                    writer.writerow([DISPLAY[variant], attack, point["intensity"], point["similarity_mean"],
                                     point["similarity_q05"], point.get("auc"), point.get("eer"),
                                     point.get("remaining_semantic_node_ratio_mean"),
                                     point.get("remaining_coordinate_node_ratio_mean"),
                                     threshold.get("tar_at_far_5pct"), threshold.get("tar_at_far_1pct")])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, attack in zip(axes.flat, SELECTED_ATTACKS):
        for variant, report in curves.items():
            if attack not in report["curves"]: continue
            rows = report["curves"][attack]
            ax.plot([row["intensity"] for row in rows], [row["similarity_mean"] for row in rows],
                    marker="o", ms=3, label=DISPLAY[variant])
        ax.set_title(attack.replace("_", " ")); ax.set_xlabel("Intensity"); ax.set_ylabel("Bit similarity")
        ax.grid(alpha=0.2); ax.set_ylim(0, 1.03)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "p0_selected_attack_curves.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    for variant, report in curves.items():
        negatives = report["protocol"].get("negative_scores", [])
        positives = [score for rows in report["curves"].values() for row in rows for score in row["scores"]]
        if positives and negatives:
            far, tar, _ = rate_curve(positives, negatives)
            ax.plot(far, tar, label=DISPLAY[variant])
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    ax.set(xlabel="False acceptance rate", ylabel="True acceptance rate", title="Pooled unseen-attack ROC")
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.savefig(output / "p0_pooled_roc.png", dpi=220); plt.close(fig)

    for variant in ("separated", "personalized"):
        fingerprints = curves.get(variant, {}).get("protocol", {}).get("clean_fingerprints", {})
        if not fingerprints: continue
        bits = np.asarray([[value == "1" for value in text] for text in fingerprints.values()], dtype=np.uint8)
        matrix = np.mean(bits[:, None, :] == bits[None, :, :], axis=2)
        fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
        image = ax.imshow(matrix, vmin=0.45, vmax=1.0, cmap="viridis")
        ax.set(title=f"{DISPLAY[variant]} clean fingerprint similarity", xlabel="Registered model", ylabel="Registered model")
        fig.colorbar(image, ax=ax, label="Bit similarity")
        fig.savefig(output / f"p0_{variant}_uniqueness_matrix.png", dpi=220); plt.close(fig)

    traditional = results / "multicity_all_baselines.json"
    if traditional.exists():
        baseline = load(traditional)
        with (output / "p0_traditional_baselines.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["method", "fidelity", "bits", "negative_mean", "negative_q95",
                             "negative_max", "runtime_ms_mean"])
            for name, method in baseline["methods"].items():
                writer.writerow([name, method["fidelity"], method["fingerprint_bits"], method["negative_mean"],
                                 method["negative_q95"], method["negative_maximum"], method["runtime_ms_mean"]])
    print(json.dumps({"variants": list(curves), "output": str(output)}))


if __name__ == "__main__":
    main()
