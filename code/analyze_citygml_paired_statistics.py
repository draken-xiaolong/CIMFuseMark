#!/usr/bin/env python3
"""Paired 64-tile statistics for native CityGML watermark baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def last_scores(curves):
    return {attack: np.asarray(max(rows, key=lambda row: float(row["intensity"]))["scores"], dtype=float)
            for attack, rows in curves.items()}


def interval(values, rng, samples=10000):
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True); parser.add_argument("--jiang", required=True)
    parser.add_argument("--embedded", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026); args = parser.parse_args()
    ours_report, jiang_report, embedded_report = load(args.ours), load(args.jiang), load(args.embedded)
    ours = last_scores(ours_report["curves"])
    baselines = {"jiang18": last_scores(jiang_report["methods"]["jiang18_citygml_radial_histogram"]["curves"])}
    baselines.update({name: last_scores(report["curves"]) for name, report in embedded_report["methods"].items()})
    rng = np.random.default_rng(args.seed); rows = []
    for baseline, values in baselines.items():
        attacks = sorted(set(ours) & set(values))
        per_attack = []
        for attack in attacks:
            difference = ours[attack] - values[attack]
            result = wilcoxon(difference, alternative="greater", zero_method="zsplit", method="approx")
            row = {"baseline": baseline, "attack": attack, "ours_mean": float(ours[attack].mean()),
                   "baseline_mean": float(values[attack].mean()), "mean_difference": float(difference.mean()),
                   "difference_ci95": interval(difference, rng), "wilcoxon_p_one_sided": float(result.pvalue)}
            rows.append(row); per_attack.append(difference)
        macro = np.mean(np.stack(per_attack), axis=0)
        result = wilcoxon(macro, alternative="greater", zero_method="zsplit", method="approx")
        rows.append({"baseline": baseline, "attack": "macro_max_intensity", "ours_mean": float(np.mean([
                     ours[a].mean() for a in attacks])), "baseline_mean": float(np.mean([values[a].mean() for a in attacks])),
                     "mean_difference": float(macro.mean()), "difference_ci95": interval(macro, rng),
                     "wilcoxon_p_one_sided": float(result.pvalue)})
    # Holm correction across the five primary macro comparisons.
    macro_rows = [row for row in rows if row["attack"] == "macro_max_intensity"]
    ordered = sorted(enumerate(macro_rows), key=lambda pair: pair[1]["wilcoxon_p_one_sided"])
    adjusted = [0.0] * len(macro_rows); running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(macro_rows)-rank) * row["wilcoxon_p_one_sided"]))
        adjusted[index] = running
    for row, value in zip(macro_rows, adjusted): row["holm_p"] = value
    output = {"protocol": {"paired_models": len(next(iter(ours.values()))), "bootstrap_samples": 10000,
                            "alternative": "CIMFuseMark-Lite > baseline", "comparison_point": "maximum intensity"},
              "rows": rows}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(macro_rows, indent=2))


if __name__ == "__main__": main()
