#!/usr/bin/env python3
"""Evaluate the v0.3 multi-relational graph baseline with real XML attacks."""

from __future__ import annotations

import itertools
import argparse
import json
import statistics
import tempfile
from pathlib import Path

from cimfusemark import attack_citygml_xml, build_citygml_graph, graph_fingerprint, similarity
from run_benchmark import auc_from_scores, eer_from_scores, quantile

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
MANIFEST = DATA_ROOT / "benchmark_manifest.json"
OUTPUT = ROOT / "results" / "graph_benchmark_results.json"


def score_summary(positive: list[float], negative: list[float]) -> dict:
    """Summarize a verification protocol when both classes are present."""
    if not positive or not negative:
        return {
            "positive_pairs": len(positive), "negative_pairs": len(negative),
            "same_source": None, "different_source": None, "verification": None,
        }
    return {
        "positive_pairs": len(positive), "negative_pairs": len(negative),
        "same_source": {
            "mean": statistics.fmean(positive),
            "q05": quantile(positive, 0.05), "minimum": min(positive),
        },
        "different_source": {
            "mean": statistics.fmean(negative),
            "q95": quantile(negative, 0.95), "maximum": max(negative),
        },
        "verification": {
            "auc": auc_from_scores(positive, negative),
            **eer_from_scores(positive, negative),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--attacks", help="Comma-separated subset of the default XML attacks")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    models = []
    for item in manifest["models"]:
        path = (DATA_ROOT / item["path"]).resolve()
        graph = build_citygml_graph(path)
        models.append({**item, "path_obj": path, "graph": graph, "fingerprint": graph_fingerprint(graph)})

    attacks = [
        ("translation", 0.0), ("scale", 0.0), ("rotation_z", 0.0),
        ("quantization", 0.001), ("object_reorder", 0.0),
        ("attribute_delete", 0.10), ("object_delete", 0.05),
    ]
    if args.attacks:
        requested = {name.strip() for name in args.attacks.split(",") if name.strip()}
        known = {name for name, _ in attacks}
        if unknown := requested - known:
            parser.error(f"unknown attacks: {', '.join(sorted(unknown))}")
        attacks = [(name, severity) for name, severity in attacks if name in requested]
    positive_records = []
    with tempfile.TemporaryDirectory(prefix="cimfusemark_xml_") as temporary:
        temporary_root = Path(temporary)
        for model in models:
            for attack, severity in attacks:
                attacked_path = temporary_root / f"{model['id']}__{attack}.gml"
                mutation = attack_citygml_xml(model["path_obj"], attacked_path, attack, severity)
                attacked_graph = build_citygml_graph(attacked_path)
                score = similarity(model["fingerprint"], graph_fingerprint(attacked_graph))
                positive_records.append({
                    "model": model["id"], "attack": attack, "severity": severity,
                    "changed_elements": mutation["changed_elements"],
                    "nodes_before": len(model["graph"].nodes), "nodes_after": len(attacked_graph.nodes),
                    "similarity": score,
                })

    negative_records, related_records = [], []
    for left, right in itertools.combinations(models, 2):
        record = {"left": left["id"], "right": right["id"],
                  "similarity": similarity(left["fingerprint"], right["fingerprint"])}
        (related_records if left["family"] == right["family"] else negative_records).append(record)
    positive = [record["similarity"] for record in positive_records]
    negative = [record["similarity"] for record in negative_records]
    related = [record["similarity"] for record in related_records]
    eer = eer_from_scores(positive, negative)
    model_splits = {model["id"]: model.get("split") for model in models}
    split_names = sorted({split for split in model_splits.values() if split})
    split_metrics = {}
    for split in split_names:
        split_positive = [record["similarity"] for record in positive_records
                          if model_splits[record["model"]] == split]
        split_negative = [record["similarity"] for record in negative_records
                          if model_splits[record["left"]] == split
                          and model_splits[record["right"]] == split]
        split_metrics[split] = score_summary(split_positive, split_negative)
    report = {
        "protocol": {
            "graph_schema": "cimfusemark_multirel_graph_v1", "models": len(models),
            "positive_pairs": len(positive), "negative_pairs": len(negative),
            "related_pairs": len(related), "xml_attacks": [name for name, _ in attacks],
        },
        "same_source": {"mean": statistics.fmean(positive), "q05": quantile(positive, 0.05), "minimum": min(positive)},
        "different_family": {"mean": statistics.fmean(negative), "q95": quantile(negative, 0.95), "maximum": max(negative)},
        "verification": {"auc": auc_from_scores(positive, negative), **eer},
        "related_versions": {
            "mean": statistics.fmean(related) if related else None,
            "minimum": min(related) if related else None,
            "scores": related_records,
        },
        "splits": split_metrics,
        "attacks": positive_records, "different_family_pairs": negative_records,
        "warning": "Exploratory non-learned baseline; thresholds require independent regional validation.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "same_source", "different_family", "verification", "related_versions", "splits", "warning")}, indent=2))


if __name__ == "__main__":
    main()
