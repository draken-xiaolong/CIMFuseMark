#!/usr/bin/env python3
"""Select the personalized-bit fraction on validation results only.

The chosen fraction minimizes clean inter-model NC while retaining validation
core-attack mean NC within ``max_robustness_drop`` of the unpersonalized model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--personalized", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .1, .25, .5, 1])
    parser.add_argument("--max-robustness-drop", type=float, default=.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attack-cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fraction in args.fractions:
        label = f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        checkpoint = output_dir / f"validation_hybrid_{label}.pt"
        result = output_dir / f"validation_hybrid_{label}_core.json"
        run([sys.executable, str(ROOT / "blend_hash_projection.py"),
             "--base", args.base, "--personalized", args.personalized,
             "--personalized-fraction", str(fraction), "--output", str(checkpoint)])
        command = [sys.executable, str(ROOT / "evaluate_robustness_curves.py"),
                   "--manifest", args.manifest, "--checkpoint", str(checkpoint),
                   "--split", "validation", "--profile", "core", "--device", args.device,
                   "--output", str(result)]
        if args.attack_cache:
            command.extend(["--attack-cache", args.attack_cache])
        run(command)
        payload = json.loads(result.read_text(encoding="utf-8"))
        point_means = [point["similarity_mean"] for points in payload["curves"].values()
                       for point in points]
        rows.append({
            "fraction": fraction,
            "personalized_bits": round(payload["protocol"]["fingerprint_bits"] * fraction),
            "robustness_mean": statistics.fmean(point_means),
            "negative_mean": payload["protocol"]["negative_mean"],
            "negative_q95": payload["protocol"]["negative_q95"],
            "negative_maximum": payload["protocol"]["negative_maximum"],
            "distinct_fingerprints": payload["protocol"]["distinct_fingerprints"],
            "checkpoint": str(checkpoint), "result": str(result),
        })

    base = min(rows, key=lambda row: abs(row["fraction"]))
    floor = base["robustness_mean"] - args.max_robustness_drop
    feasible = [row for row in rows if row["robustness_mean"] >= floor]
    selected = min(feasible, key=lambda row: (row["negative_mean"], -row["robustness_mean"], row["fraction"]))
    report = {
        "protocol": {
            "selection_split": "validation", "test_results_used": False,
            "criterion": "minimum clean negative mean NC subject to core-attack mean NC retention",
            "max_robustness_drop": args.max_robustness_drop,
            "base_robustness_mean": base["robustness_mean"], "robustness_floor": floor,
            "core_points": 13, "fractions_fixed_before_evaluation": args.fractions,
        },
        "candidates": rows, "selected": selected,
    }
    target = output_dir / "hybrid_fraction_selection.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
