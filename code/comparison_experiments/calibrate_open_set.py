#!/usr/bin/env python3
"""Add validation-only open-set thresholds to an existing baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comparison_experiments.baselines import all_baselines, bit_similarity
from run_benchmark import quantile

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--registered-split", default="test")
    parser.add_argument("--calibration-split", default="validation")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report_path = Path(args.results); report = json.loads(report_path.read_text(encoding="utf-8"))
    registered = [item for item in manifest["models"] if item.get("split") == args.registered_split]
    calibration = [item for item in manifest["models"] if item.get("split") == args.calibration_split]
    for method in all_baselines():
        if method.name not in report["methods"]: continue
        registered_bits = [method.fingerprint((DATA_ROOT / item["path"]).resolve()) for item in registered]
        calibration_bits = [method.fingerprint((DATA_ROOT / item["path"]).resolve()) for item in calibration]
        maxima = [max(bit_similarity(bits, candidate) for candidate in registered_bits)
                  for bits in calibration_bits]
        thresholds = {"far_5pct": quantile(maxima, 0.95), "far_1pct": quantile(maxima, 0.99)}
        method_report = report["methods"][method.name]
        method_report["open_set"] = {
            "registered_split": args.registered_split, "calibration_split": args.calibration_split,
            "registered_models": len(registered), "calibration_models": len(calibration),
            "threshold_rule": "validation impostor maximum-match quantile", "thresholds": thresholds,
            "observed_far": {name: sum(score >= threshold for score in maxima) / len(maxima)
                             for name, threshold in thresholds.items()}, "impostor_maxima": maxima,
        }
        for points in method_report["curves"].values():
            for point in points:
                scores = point["scores"]
                point["tar_at_far_5pct"] = sum(score >= thresholds["far_5pct"] for score in scores) / len(scores)
                point["tar_at_far_1pct"] = sum(score >= thresholds["far_1pct"] for score in scores) / len(scores)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: value.get("open_set", {}).get("thresholds")
                      for name, value in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
