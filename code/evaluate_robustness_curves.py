#!/usr/bin/env python3
"""Evaluate a trained CIMFuseMark checkpoint over attack-intensity sweeps."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import torch

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors
from run_benchmark import quantile

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"

DEFAULT_SWEEPS = {
    "object_delete": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "building_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "surface_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "attribute_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "quantization": [0.001, 0.005, 0.01, 0.02, 0.05],
    "rotation_z": [30.0, 60.0, 90.0, 120.0, 180.0],
}


def bit_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left == right).float().mean().item())


def encode(model, graph, relations, device, relation_mode, feature_mode):
    x, edge_index, edge_type = graph_tensors(graph, relations, device)
    if relation_mode == "no_edges":
        edge_index, edge_type = edge_index[:, :0], edge_type[:0]
    if feature_mode == "geometry":
        x = x.clone(); x[:, 8:] = 0.0
    return model.fingerprint(model.encode(x, edge_index, edge_type))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=str(ROOT / "results" / "robustness_curves.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attacks", nargs="+", choices=tuple(DEFAULT_SWEEPS),
                        help="Only evaluate selected attack families")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    relations = checkpoint["relations"]
    relation_mode = checkpoint.get("relation_mode", "typed")
    feature_mode = checkpoint.get("feature_mode", "full")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = [item for item in manifest["models"] if item.get("split") == args.split]
    if not selected:
        raise ValueError(f"No models in split {args.split!r}")

    clean_graphs = {}
    for item in selected:
        clean_graphs[item["id"]] = build_citygml_graph((DATA_ROOT / item["path"]).resolve())
    input_dim = len(next(iter(clean_graphs.values())).nodes[0].features)
    model = CIMFuseRGCN(
        input_dim, int(config["hidden_dim"]), int(config["embedding_dim"]),
        max(relations.values(), default=0) + 1, int(config["fingerprint_bits"]), int(config["seed"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        clean_bits = {item["id"]: encode(model, clean_graphs[item["id"]], relations, device,
                                                relation_mode, feature_mode)
                      for item in selected}
        curves = {}
        with tempfile.TemporaryDirectory(prefix="cimfusemark_curves_") as temporary:
            temporary_root = Path(temporary)
            selected_sweeps = {name: DEFAULT_SWEEPS[name] for name in args.attacks} if args.attacks else DEFAULT_SWEEPS
            for attack, levels in selected_sweeps.items():
                points = []
                for level in levels:
                    scores, changed, node_ratios, valid = [], [], [], 0
                    for item in selected:
                        source = (DATA_ROOT / item["path"]).resolve()
                        attacked_path = temporary_root / f"{item['id']}__{attack}_{level}.gml"
                        mutation = attack_citygml_xml(source, attacked_path, attack, level,
                                                      seed=int(config["seed"]))
                        try:
                            attacked = build_citygml_graph(attacked_path)
                            bits = encode(model, attacked, relations, device, relation_mode, feature_mode)
                            scores.append(bit_similarity(clean_bits[item["id"]], bits))
                            node_ratios.append(len(attacked.nodes) / max(len(clean_graphs[item["id"]].nodes), 1))
                            valid += 1
                        except ValueError:
                            # A destructively emptied model is an authentication failure, not a missing sample.
                            scores.append(0.0)
                            node_ratios.append(0.0)
                        changed.append(int(mutation["changed_elements"]))
                    points.append({
                        "intensity": level, "models": len(scores),
                        "similarity_mean": statistics.fmean(scores),
                        "similarity_q05": quantile(scores, 0.05),
                        "similarity_minimum": min(scores),
                        "ber_mean": 1.0 - statistics.fmean(scores),
                        "valid_graph_rate": valid / len(scores),
                        "remaining_node_ratio_mean": statistics.fmean(node_ratios),
                        "remaining_node_ratio_minimum": min(node_ratios),
                        "changed_elements_mean": statistics.fmean(changed),
                        "scores": scores,
                    })
                curves[attack] = points

    output = {
        "protocol": {
            "manifest": str(args.manifest), "checkpoint": str(args.checkpoint),
            "split": args.split, "models": len(selected), "device": str(device),
            "relation_mode": relation_mode, "feature_mode": feature_mode,
            "fingerprint_bits": int(config["fingerprint_bits"]),
        },
        "curves": curves,
        "warning": "Attack-specific similarity curves; thresholds are not recalibrated at each intensity.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {attack: [{"intensity": point["intensity"],
                         "mean": round(point["similarity_mean"], 6),
                         "q05": round(point["similarity_q05"], 6)} for point in points]
               for attack, points in curves.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
