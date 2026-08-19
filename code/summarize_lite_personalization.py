#!/usr/bin/env python3
"""Summarize Base, projection-only, and partial-encoder Lite personalization."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

VARIANTS = {
    "base": ("Lite Base", "compression/rgcn_h64_e192_geometry_depth_seed2026"),
    "projection": ("Lite + projection", "personalized_lite/cimfusemark_lite_personalized_seed2026"),
    "finetune": ("Lite + partial encoder fine-tuning", "personalized_lite_finetune/cimfusemark_lite_finetuned_seed2026"),
}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(args.results); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows, attack_rows = [], []
    for variant, (label, stem) in VARIANTS.items():
        curves = json.loads((root/f"{stem}_core.json").read_text(encoding="utf-8"))
        opened = json.loads((root/f"{stem}_open_set.json").read_text(encoding="utf-8"))
        points = [point for values in curves["curves"].values() for point in values]
        fixed = [point for values in opened["fixed_threshold_attacks"].values() for point in values]
        protocol = curves["protocol"]; open_set = opened["open_set"]
        rows.append({
            "variant": variant, "label": label, "negative_mean": protocol["negative_mean"],
            "negative_q95": protocol["negative_q95"], "negative_max": protocol["negative_maximum"],
            "auc_mean": statistics.fmean(point["auc"] for point in points),
            "eer_mean": statistics.fmean(point["eer"] for point in points),
            "tar_far5_mean": statistics.fmean(point["tar_at_far_5pct"] for point in fixed),
            "tar_far1_mean": statistics.fmean(point["tar_at_far_1pct"] for point in fixed),
            "far5_threshold": open_set["thresholds"]["far_5pct"],
            "observed_far5": open_set["observed_open_set_far_by_target"]["far_5pct"],
        })
        for attack, values in opened["fixed_threshold_attacks"].items():
            for point in values:
                attack_rows.append({"variant": variant, "attack": attack, "intensity": point["intensity"],
                                    "tar_far5": point["tar_at_far_5pct"], "tar_far1": point["tar_at_far_1pct"]})
    with (output/"lite_personalization_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output/"lite_personalization_attacks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attack_rows[0])); writer.writeheader(); writer.writerows(attack_rows)
    (output/"lite_personalization_summary.json").write_text(json.dumps({"experiments": rows}, indent=2), encoding="utf-8")
    lines = ["# CIMFuseMark-Lite personalization exploration", "",
             "| Variant | Negative mean | Negative q95 | Negative max | AUC | EER | TAR@FAR=5% |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['label']} | {row['negative_mean']:.4f} | {row['negative_q95']:.4f} | "
                     f"{row['negative_max']:.4f} | {row['auc_mean']:.4f} | {row['eer_mean']:.4f} | "
                     f"{row['tar_far5_mean']:.4f} |")
    lines += ["", "## Selection", "",
              "Partial-encoder fine-tuning is the best personalized Lite variant for seed 2026. It preserves the projection-only q95, improves AUC/EER, and recovers most of the Base open-set TAR.",
              "The result remains exploratory until repeated with additional seeds. The Base model remains the non-transductive reference.", ""]
    (output/"LITE_PERSONALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
