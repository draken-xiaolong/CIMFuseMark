#!/usr/bin/env python3
"""Aggregate authentication curves from independently trained seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); runs = [json.loads(Path(path).read_text()) for path in args.inputs]
    summary = {"runs": args.inputs, "seeds": [run["protocol"].get("checkpoint") for run in runs], "curves": {}}
    common_attacks = set.intersection(*(set(run["curves"]) for run in runs))
    for attack in sorted(common_attacks):
        summary["curves"][attack] = []
        levels = [point["intensity"] for point in runs[0]["curves"][attack]]
        for level in levels:
            points = [next(point for point in run["curves"][attack] if point["intensity"] == level) for run in runs]
            row = {"intensity": level}
            for metric in ("similarity_mean", "similarity_q05", "auc", "eer", "frr_at_negative_q95"):
                values = [float(point[metric]) for point in points]
                row[metric + "_mean"] = statistics.fmean(values)
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                row[metric + "_values"] = values
            summary["curves"][attack].append(row)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "runs": len(runs), "attacks": sorted(common_attacks)}))


if __name__ == "__main__": main()
