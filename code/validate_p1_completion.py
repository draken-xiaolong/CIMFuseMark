#!/usr/bin/env python3
"""Fail closed unless every planned P1 artifact is present and protocol-compliant."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_points(report: dict) -> bool:
    keys = ("similarity_mean", "similarity_q05", "auc", "eer")
    return all(math.isfinite(float(point[key])) for rows in report["curves"].values()
               for point in rows for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(args.results)
    experiments = load(Path(args.matrix))["experiments"]; rows = []; failures = []
    for experiment in experiments:
        identifier = experiment["id"]; reuse = experiment.get("reuse")
        checkpoint = root / f"{reuse or 'p1_' + identifier}.pt"
        curves = root / f"p1_{identifier}_core_curves.json"
        opened = root / f"p1_{identifier}_open_set.json"
        missing = [str(path) for path in (checkpoint, curves, opened) if not path.exists()]
        if missing:
            failures.append({"id": identifier, "error": "missing", "paths": missing}); continue
        curve_report = load(curves); open_report = load(opened)
        errors = []
        if curve_report["protocol"].get("models") != 64: errors.append("core evaluation does not contain 64 test models")
        if len(curve_report["curves"]) != 10: errors.append("core profile does not contain 10 attack families")
        if sum(len(points) for points in curve_report["curves"].values()) != 13:
            errors.append("core profile does not contain 13 attack-strength points")
        if not finite_points(curve_report): errors.append("non-finite core metric")
        if set(curve_report["curves"]) != set(open_report["fixed_threshold_attacks"]):
            errors.append("core/open-set attack keys differ")
        if "far_5pct" not in open_report["open_set"].get("thresholds", {}): errors.append("missing frozen FAR=5% threshold")
        training = root / f"{reuse or 'p1_' + identifier}_training.json"
        if not training.exists(): errors.append("missing training report")
        else:
            training_report = load(training)
            if int(training_report["config"].get("epochs", 0)) != 800: errors.append("training epochs != 800")
            expected_seed = int(experiment.get("seed", 2026))
            if int(training_report["config"].get("seed", -1)) != expected_seed: errors.append("training seed mismatch")
        row = {"id": identifier, "group": experiment["group"], "checkpoint": str(checkpoint),
               "curves": str(curves), "open_set": str(opened), "status": "failed" if errors else "complete"}
        if errors: row["errors"] = errors; failures.append(row)
        rows.append(row)
    required_global = (root / "p1_efficiency.json",)
    for path in required_global:
        if not path.exists(): failures.append({"error": "missing global artifact", "path": str(path)})
    report = {"status": "complete" if not failures and len(rows) == len(experiments) else "failed",
              "planned": len(experiments), "validated": len(rows), "failures": failures, "experiments": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "planned": len(experiments), "failures": len(failures)}))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
