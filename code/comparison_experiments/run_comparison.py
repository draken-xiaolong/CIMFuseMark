#!/usr/bin/env python3
"""Run traditional zero-watermark baselines under the shared CIM attack protocol."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from cimfusemark import attack_citygml_xml  # noqa: E402
from comparison_experiments.baselines import all_baselines, bit_similarity  # noqa: E402
from evaluate_robustness_curves import DEFAULT_SWEEPS  # noqa: E402
from run_benchmark import auc_from_scores, eer_from_scores, quantile  # noqa: E402

DATA_ROOT = CODE_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DATA_ROOT / "plateau_manifest.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+", choices=tuple(DEFAULT_SWEEPS))
    parser.add_argument("--methods", nargs="+", help="Only run selected baseline identifiers")
    parser.add_argument("--output", default=str(CODE_ROOT / "results" / "traditional_baseline_comparison.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--work-root", help="Directory for temporary attacked CityGML files")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = [item for item in manifest["models"] if item.get("split") == args.split]
    if len(items) < 2: raise ValueError("Comparison requires at least two models")
    sweeps = {name: DEFAULT_SWEEPS[name] for name in args.attacks} if args.attacks else DEFAULT_SWEEPS

    output = {"protocol": {"manifest": args.manifest, "split": args.split, "models": len(items),
                           "seed": args.seed, "attacks": sweeps,
                           "note": "Mesh-only papers are reported as explicit CityGML adaptations."},
              "methods": {}}
    work_root = Path(args.work_root).resolve() if args.work_root else None
    if work_root: work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cim_baseline_comparison_", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        attack_paths = {}
        for attack, levels in sweeps.items():
            for level in levels:
                for item_index, item in enumerate(items):
                    source = (DATA_ROOT / item["path"]).resolve()
                    target = temporary_root / f"{item['id']}__{attack}_{level}.gml"
                    mutation = attack_citygml_xml(source, target, attack, level, seed=args.seed)
                    attack_paths[(attack, level, item["id"])] = (target, mutation)

        methods = all_baselines()
        if args.methods:
            methods = [method for method in methods if method.name in set(args.methods)]
        for method in methods:
            clean, runtimes = {}, []
            for item in items:
                bits, elapsed = method.timed_fingerprint((DATA_ROOT / item["path"]).resolve())
                clean[item["id"]] = bits; runtimes.append(elapsed)
            negatives = [bit_similarity(clean[left["id"]], clean[right["id"]])
                         for left, right in itertools.combinations(items, 2)]
            threshold = quantile(negatives, 0.95)
            curves = {}
            for attack, levels in sweeps.items():
                points = []
                for level in levels:
                    positives, changed = [], []
                    for item in items:
                        path, mutation = attack_paths[(attack, level, item["id"])]
                        try:
                            bits, elapsed = method.timed_fingerprint(path)
                            positives.append(bit_similarity(clean[item["id"]], bits)); runtimes.append(elapsed)
                        except ValueError:
                            positives.append(0.0)
                        changed.append(int(mutation["changed_elements"]))
                    points.append({"intensity": level, "positive_mean": statistics.fmean(positives),
                                   "positive_q05": quantile(positives, 0.05), "positive_minimum": min(positives),
                                   "auc": auc_from_scores(positives, negatives), **eer_from_scores(positives, negatives),
                                   "frr_at_negative_q95": sum(score < threshold for score in positives) / len(positives),
                                   "scores": positives, "changed_elements_mean": statistics.fmean(changed)})
                curves[attack] = points
            output["methods"][method.name] = {
                "citation": method.citation, "fidelity": method.fidelity, "fingerprint_bits": method.bits,
                "negative_mean": statistics.fmean(negatives), "negative_q95": threshold,
                "negative_maximum": max(negatives), "runtime_ms_mean": statistics.fmean(runtimes) * 1000,
                "curves": curves,
            }
            print(json.dumps({"method": method.name, "bits": method.bits,
                              "negative_mean": output["methods"][method.name]["negative_mean"],
                              "negative_q95": threshold, "negative_maximum": max(negatives)}))
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target)}))


if __name__ == "__main__":
    main()
