#!/usr/bin/env python3
"""Evaluate same-source robustness and different-family discrimination."""

from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

from cimfusemark import attack_points, attack_semantics, extract_citygml, fingerprint, similarity

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
MANIFEST = DATA_ROOT / "benchmark_manifest.json"
OUTPUT = ROOT / "results" / "benchmark_results.json"


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def auc_from_scores(positive: list[float], negative: list[float]) -> float:
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return wins / (len(positive) * len(negative))


def eer_from_scores(positive: list[float], negative: list[float]) -> dict[str, float]:
    thresholds = sorted(set(positive + negative), reverse=True)
    best = None
    for threshold in thresholds:
        fnr = sum(score < threshold for score in positive) / len(positive)
        fpr = sum(score >= threshold for score in negative) / len(negative)
        candidate = (abs(fnr - fpr), (fnr + fpr) / 2, threshold, fnr, fpr)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {"eer": best[1], "threshold": best[2], "fnr": best[3], "fpr": best[4]}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = []
    failures = []
    for item in manifest["models"]:
        path = (DATA_ROOT / item["path"]).resolve()
        try:
            points, semantics = extract_citygml(path)
        except (ValueError, OSError) as exc:
            failures.append({"id": item["id"], "error": str(exc)})
            continue
        models.append({
            **item,
            "points": points,
            "semantics": semantics,
            "fingerprint": fingerprint(points, semantics),
        })

    positive_records = []
    attack_specs = [
        ("translation", 0.0), ("scale", 0.0), ("rotation_3d", 0.0),
        ("noise", 0.002), ("quantization", 0.001), ("spatial_crop", 0.05),
    ]
    for model in models:
        for attack, severity in attack_specs:
            attacked_points = attack_points(model["points"], attack, severity)
            attacked_fp = fingerprint(attacked_points, model["semantics"])
            positive_records.append({
                "model": model["id"], "attack": attack, "severity": severity,
                "similarity": similarity(model["fingerprint"], attacked_fp),
            })
        attacked_semantics = attack_semantics(model["semantics"], 0.10)
        attacked_fp = fingerprint(model["points"], attacked_semantics)
        positive_records.append({
            "model": model["id"], "attack": "semantic_object_deletion", "severity": 0.10,
            "similarity": similarity(model["fingerprint"], attacked_fp),
        })

    negative_records = []
    related_records = []
    for left, right in itertools.combinations(models, 2):
        record = {
            "left": left["id"], "right": right["id"],
            "similarity": similarity(left["fingerprint"], right["fingerprint"]),
        }
        if left["family"] == right["family"]:
            related_records.append(record)
        else:
            negative_records.append(record)

    positive = [record["similarity"] for record in positive_records]
    negative = [record["similarity"] for record in negative_records]
    eer = eer_from_scores(positive, negative)
    related = [record["similarity"] for record in related_records]
    false_accepts = [record for record in negative_records if record["similarity"] >= eer["threshold"]]
    report = {
        "protocol": {
            "models_loaded": len(models), "models_failed": failures,
            "positive_pairs": len(positive), "negative_pairs": len(negative),
            "related_cross_version_pairs_excluded_from_negatives": len(related_records),
        },
        "same_source": {
            "mean": statistics.fmean(positive), "q05": quantile(positive, 0.05),
            "minimum": min(positive),
        },
        "different_family": {
            "mean": statistics.fmean(negative), "q95": quantile(negative, 0.95),
            "maximum": max(negative),
        },
        "verification": {"auc": auc_from_scores(positive, negative), **eer},
        "related_versions": {
            "pairs": len(related),
            "mean": statistics.fmean(related) if related else None,
            "minimum": min(related) if related else None,
        },
        "observed_false_accepts_at_eer_threshold": false_accepts,
        "attacks": positive_records,
        "different_family_pairs": negative_records,
        "related_pairs": related_records,
        "warning": manifest["caveat"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "protocol", "same_source", "different_family", "verification",
        "related_versions", "observed_false_accepts_at_eer_threshold", "warning"
    )}, indent=2))


if __name__ == "__main__":
    main()
