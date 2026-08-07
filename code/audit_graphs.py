#!/usr/bin/env python3
"""Audit every benchmark file for graph construction coverage and invariants."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cimfusemark import build_citygml_graph

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def main() -> None:
    manifest = json.loads((DATA_ROOT / "benchmark_manifest.json").read_text(encoding="utf-8"))
    records, failures = [], []
    for item in manifest["models"]:
        try:
            graph = build_citygml_graph((DATA_ROOT / item["path"]).resolve())
            invalid_edges = sum(edge.source >= len(graph.nodes) or edge.target >= len(graph.nodes) for edge in graph.edges)
            orphan_count = sum(node.parent is None for node in graph.nodes)
            records.append({
                "id": item["id"], "nodes": len(graph.nodes), "edges": len(graph.edges),
                "roots": orphan_count, "invalid_edges": invalid_edges,
                "node_types": dict(Counter(node.node_type for node in graph.nodes)),
                "relation_types": dict(Counter(edge.relation for edge in graph.edges)),
            })
        except Exception as exc:
            failures.append({"id": item["id"], "error": f"{type(exc).__name__}: {exc}"})
    report = {"models": len(records), "failures": failures, "records": records}
    output = ROOT / "results" / "graph_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"models": len(records), "failures": failures,
                      "total_nodes": sum(record["nodes"] for record in records),
                      "total_edges": sum(record["edges"] for record in records)}, indent=2))
    if failures or any(record["invalid_edges"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

