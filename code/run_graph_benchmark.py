#!/usr/bin/env python3
"""Evaluate the v0.3 multi-relational graph baseline with real XML attacks."""

from __future__ import annotations

import itertools
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


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
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
    report = {
        "protocol": {
            "graph_schema": "cimfusemark_multirel_graph_v1", "models": len(models),
            "positive_pairs": len(positive), "negative_pairs": len(negative),
            "related_pairs": len(related), "xml_attacks": [name for name, _ in attacks],
        },
        "same_source": {"mean": statistics.fmean(positive), "q05": quantile(positive, 0.05), "minimum": min(positive)},
        "different_family": {"mean": statistics.fmean(negative), "q95": quantile(negative, 0.95), "maximum": max(negative)},
        "verification": {"auc": auc_from_scores(positive, negative), **eer},
        "related_versions": {"mean": statistics.fmean(related), "minimum": min(related), "scores": related_records},
        "attacks": positive_records, "different_family_pairs": negative_records,
        "warning": "Non-learned graph baseline on a small, non-independent standards-example corpus.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("protocol", "same_source", "different_family", "verification", "related_versions", "warning")}, indent=2))


if __name__ == "__main__":
    main()

