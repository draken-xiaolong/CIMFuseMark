#!/usr/bin/env python3
"""Evaluate native CityGML embedded watermarking methods on shared attacks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from cimfusemark import attack_citygml_xml  # noqa: E402
from comparison_experiments.citygml_embedded import all_embedded_methods, payload  # noqa: E402
from evaluate_robustness_curves import DEFAULT_SWEEPS  # noqa: E402

DATA_ROOT = CODE_ROOT / "data"


def similarity(left, right):
    return float((left == right).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DATA_ROOT / "plateau_manifest.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+", choices=tuple(DEFAULT_SWEEPS))
    parser.add_argument("--methods", nargs="+", help="Only run selected method identifiers")
    parser.add_argument("--output", default=str(CODE_ROOT / "results" / "citygml_embedded_comparison.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--work-root")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = [item for item in manifest["models"] if item.get("split") == args.split]
    sweeps = {k: DEFAULT_SWEEPS[k] for k in args.attacks} if args.attacks else DEFAULT_SWEEPS
    watermark = payload(args.seed)
    output = {"protocol": {"manifest": args.manifest, "split": args.split, "models": len(items),
                           "seed": args.seed, "payload_bits": len(watermark), "attacks": sweeps,
                           "metric": "binary NC = 1 - BER"}, "methods": {}}
    work_root = Path(args.work_root).resolve() if args.work_root else None
    if work_root: work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cim_embedded_", dir=work_root) as temporary:
        temp = Path(temporary)
        methods = all_embedded_methods()
        if args.methods:
            methods = [method for method in methods if method.name in set(args.methods)]
        for method in methods:
            watermarked, clean_scores = {}, []
            for item in items:
                source = (DATA_ROOT / item["path"]).resolve()
                target = temp / method.name / "clean" / f"{item['id']}.gml"
                method.embed(source, target, watermark); watermarked[item["id"]] = target
                clean_scores.append(similarity(watermark, method.extract(target)))
            curves = {}
            for attack, levels in sweeps.items():
                points = []
                for level in levels:
                    scores = []
                    for item in items:
                        target = temp / method.name / attack / f"{item['id']}__{level}.gml"
                        try:
                            attack_citygml_xml(watermarked[item["id"]], target, attack, level, seed=args.seed)
                            scores.append(similarity(watermark, method.extract(target)))
                        except (ValueError, ET.ParseError):
                            scores.append(0.0)
                    points.append({"intensity": level, "similarity_mean": statistics.fmean(scores),
                                   "ber_mean": 1-statistics.fmean(scores), "scores": scores})
                curves[attack] = points
            output["methods"][method.name] = {"citation": method.citation, "fidelity": method.fidelity,
                                              "embedded": True, "fingerprint_bits": method.bits,
                                              "clean_nc_mean": statistics.fmean(clean_scores), "curves": curves}
            print(json.dumps({"method": method.name, "clean_nc": statistics.fmean(clean_scores)}))
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target)}))


if __name__ == "__main__":
    # Imported here so the normal path stays lightweight.
    import xml.etree.ElementTree as ET
    main()
