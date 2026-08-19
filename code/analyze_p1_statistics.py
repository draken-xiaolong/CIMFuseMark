#!/usr/bin/env python3
"""P1 ablation summaries, multiseed bootstrap CIs and paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from run_benchmark import auc_from_scores, eer_from_scores, quantile


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point_map(report: dict) -> dict[tuple[str, float], dict]:
    return {(attack, float(point["intensity"])): point
            for attack, points in report["curves"].items() for point in points}


def fixed_map(report: dict) -> dict[tuple[str, float], dict]:
    return {(attack, float(point["intensity"])): point
            for attack, points in report["fixed_threshold_attacks"].items() for point in points}


def bootstrap_multiseed(curves: list[dict], opens: list[dict], iterations: int, seed: int) -> dict:
    rng = random.Random(seed); maps = [point_map(item) for item in curves]; fixed = [fixed_map(item) for item in opens]
    keys = sorted(set.intersection(*(set(item) for item in maps)))
    output = {}
    for key in keys:
        rows = [item[key] for item in maps]; estimates = {name: [] for name in ("auc", "eer", "tar", "far", "frr")}
        for _ in range(iterations):
            chosen = [rng.randrange(len(rows)) for _ in rows]
            positives, negatives, maxima = [], [], []
            tar_values = []
            for index in chosen:
                row = rows[index]; negative = curves[index]["protocol"]["negative_scores"]
                positives.extend(row["scores"][rng.randrange(len(row["scores"]))] for _ in row["scores"])
                negatives.extend(negative[rng.randrange(len(negative))] for _ in negative)
                threshold = opens[index]["open_set"]["thresholds"]["far_5pct"]
                sampled = [row["scores"][rng.randrange(len(row["scores"]))] for _ in row["scores"]]
                tar_values.append(sum(value >= threshold for value in sampled) / len(sampled))
                source_maxima = list(opens[index]["open_set"]["impostor_maxima"].values())
                maxima.extend(source_maxima[rng.randrange(len(source_maxima))] for _ in source_maxima)
            estimates["auc"].append(auc_from_scores(positives, negatives))
            estimates["eer"].append(eer_from_scores(positives, negatives)["eer"])
            estimates["tar"].append(statistics.fmean(tar_values)); estimates["frr"].append(1-statistics.fmean(tar_values))
            # Thresholds are frozen per seed; report the hierarchical resample of observed validation FAR.
            estimates["far"].append(statistics.fmean(
                opens[index]["open_set"]["observed_open_set_far_by_target"]["far_5pct"] for index in chosen))
        output[f"{key[0]}@{key[1]}"] = {
            metric: {"mean": statistics.fmean(values), "ci95": [quantile(values, 0.025), quantile(values, 0.975)]}
            for metric, values in estimates.items()}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(); root = Path(args.results); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    matrix = load(Path(args.matrix))["experiments"]
    summaries = []
    for item in matrix:
        curves_path = root / f"p1_{item['id']}_core_curves.json"
        open_path = root / f"p1_{item['id']}_open_set.json"
        if not curves_path.exists() or not open_path.exists(): continue
        curves = load(curves_path); opened = load(open_path); points = [p for rows in curves["curves"].values() for p in rows]
        fixed = [p for rows in opened["fixed_threshold_attacks"].values() for p in rows]
        protocol = curves["protocol"]
        summaries.append({"id": item["id"], "group": item["group"],
                          "auc_mean": statistics.fmean(p["auc"] for p in points),
                          "eer_mean": statistics.fmean(p["eer"] for p in points),
                          "similarity_mean": statistics.fmean(p["similarity_mean"] for p in points),
                          "tar_far5_mean": statistics.fmean(p["tar_at_far_5pct"] for p in fixed),
                          "negative_q95": protocol["negative_q95"], "negative_max": protocol["negative_maximum"],
                          "distinct": protocol["distinct_fingerprints"], "collisions": protocol["collision_pairs"]})
    with (output / "p1_ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)

    seed_ids = ("full_seed2026", "full_seed2027", "full_seed2028")
    seed_curves = [load(root / f"p1_{name}_core_curves.json") for name in seed_ids]
    seed_open = [load(root / f"p1_{name}_open_set.json") for name in seed_ids]
    bootstrap = bootstrap_multiseed(seed_curves, seed_open, args.bootstrap, 20260819)

    baseline = load(Path(args.baselines)); main_points = point_map(seed_curves[0]); main_fixed = fixed_map(seed_open[0])
    paired = {}
    try:
        from scipy.stats import wilcoxon
        for key, main in main_points.items():
            attack, level = key; candidates = []
            for name, method in baseline["methods"].items():
                rows = {float(row["intensity"]): row for row in method["curves"].get(attack, [])}
                if level in rows: candidates.append((rows[level]["auc"], name, method, rows[level]))
            if not candidates: continue
            _auc, name, method, row = max(candidates)
            main_threshold = seed_open[0]["open_set"]["thresholds"]["far_5pct"]
            baseline_threshold = method["open_set"]["thresholds"]["far_5pct"]
            differences = [(left-main_threshold) - (right-baseline_threshold)
                           for left, right in zip(main["scores"], row["scores"])]
            test = wilcoxon(differences, alternative="greater", zero_method="wilcox")
            paired[f"{attack}@{level}"] = {"best_baseline": name, "main_auc": main["auc"],
                                             "baseline_auc": row["auc"], "statistic": float(test.statistic),
                                             "pvalue_one_sided": float(test.pvalue),
                                             "margin_difference_mean": statistics.fmean(differences)}
    except (ImportError, ValueError) as error:
        paired["warning"] = str(error)
    report = {"experiments": summaries, "multiseed_bootstrap": bootstrap,
              "paired_wilcoxon_authentication_margin": paired,
              "bootstrap_iterations": args.bootstrap, "seed_ids": seed_ids}
    (output / "p1_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"experiments": len(summaries), "bootstrap_attacks": len(bootstrap), "paired": len(paired)}))


if __name__ == "__main__":
    main()
