#!/usr/bin/env python3
"""P1 ablation summaries, multiseed bootstrap CIs and paired tests."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from run_benchmark import quantile


def fast_auc(positive: list[float], negative: list[float]) -> float:
    """Equivalent Mann--Whitney AUC without the quadratic pair loop."""
    positives = np.asarray(positive); negatives = np.sort(np.asarray(negative))
    lower = np.searchsorted(negatives, positives, side="left")
    upper = np.searchsorted(negatives, positives, side="right")
    return float(np.sum(lower + 0.5 * (upper-lower)) / (len(positives)*len(negatives)))


def fast_eer(positive: list[float], negative: list[float]) -> float:
    """Exact discrete EER with binary searches over sorted score arrays."""
    positives, negatives = np.sort(np.asarray(positive)), np.sort(np.asarray(negative))
    thresholds = np.unique(np.concatenate((positives, negatives)))
    fnr = np.searchsorted(positives, thresholds, side="left") / len(positives)
    fpr = (len(negatives)-np.searchsorted(negatives, thresholds, side="left")) / len(negatives)
    eer = (fnr+fpr)/2; difference = np.abs(fnr-fpr)
    index = min(range(len(thresholds)), key=lambda i: (difference[i], eer[i], thresholds[i]))
    return float(eer[index])


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point_map(report: dict) -> dict[tuple[str, float], dict]:
    return {(attack, float(point["intensity"])): point
            for attack, points in report["curves"].items() for point in points}


def fixed_map(report: dict) -> dict[tuple[str, float], dict]:
    return {(attack, float(point["intensity"])): point
            for attack, points in report["fixed_threshold_attacks"].items() for point in points}


def bootstrap_multiseed(curves: list[dict], opens: list[dict], iterations: int, seed: int) -> dict:
    rng = np.random.default_rng(seed); maps = [point_map(item) for item in curves]
    keys = sorted(set.intersection(*(set(item) for item in maps)))
    output = {}
    for key in keys:
        rows = [item[key] for item in maps]; estimates = {name: [] for name in ("auc", "eer", "tar", "far", "frr")}
        for _ in range(iterations):
            chosen = rng.integers(0, len(rows), size=len(rows)).tolist()
            positives, negatives = [], []
            tar_values, far_values = [], []
            for index in chosen:
                row = rows[index]; negative = curves[index]["protocol"]["negative_scores"]
                positives.extend(rng.choice(row["scores"], size=len(row["scores"]), replace=True).tolist())
                negatives.extend(rng.choice(negative, size=len(negative), replace=True).tolist())
                threshold = opens[index]["open_set"]["thresholds"]["far_5pct"]
                sampled = rng.choice(row["scores"], size=len(row["scores"]), replace=True).tolist()
                tar_values.append(sum(value >= threshold for value in sampled) / len(sampled))
                source_maxima = list(opens[index]["open_set"]["impostor_maxima"].values())
                sampled_maxima = rng.choice(source_maxima, size=len(source_maxima), replace=True).tolist()
                far_values.append(sum(value >= threshold for value in sampled_maxima) / len(sampled_maxima))
            estimates["auc"].append(fast_auc(positives, negatives))
            estimates["eer"].append(fast_eer(positives, negatives))
            estimates["tar"].append(statistics.fmean(tar_values)); estimates["frr"].append(1-statistics.fmean(tar_values))
            # Thresholds remain frozen per seed while validation impostor maxima are resampled.
            estimates["far"].append(statistics.fmean(far_values))
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
            try:
                test = wilcoxon(differences, alternative="greater", zero_method="wilcox")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            paired[f"{attack}@{level}"] = {"best_baseline": name, "main_auc": main["auc"],
                                             "baseline_auc": row["auc"], "statistic": statistic,
                                             "pvalue_one_sided": pvalue,
                                             "margin_difference_mean": statistics.fmean(differences)}
    except ImportError as error:
        paired["warning"] = str(error)
    report = {"experiments": summaries, "multiseed_bootstrap": bootstrap,
              "paired_wilcoxon_authentication_margin": paired,
              "bootstrap_iterations": args.bootstrap, "seed_ids": seed_ids}
    (output / "p1_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"experiments": len(summaries), "bootstrap_attacks": len(bootstrap), "paired": len(paired)}))


if __name__ == "__main__":
    main()
