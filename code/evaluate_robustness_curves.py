#!/usr/bin/env python3
"""Evaluate a trained CIMFuseMark checkpoint over attack-intensity sweeps."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import statistics
import tempfile
from pathlib import Path

import torch

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import create_model, graph_tensors
from run_benchmark import auc_from_scores, eer_from_scores, quantile

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"

DEFAULT_SWEEPS = {
    "object_delete": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    "building_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "surface_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "attribute_delete": [0.10, 0.20, 0.40, 0.60, 0.80],
    "quantization": [0.001, 0.005, 0.01, 0.02, 0.05],
    "rotation_z": [30.0, 60.0, 90.0, 120.0, 180.0],
    "coordinate_noise": [0.0001, 0.0005, 0.001, 0.002, 0.005],
    "sequential": [0.10, 0.20, 0.40, 0.60],
    "lod2_to_lod1": [1.0],
    "hierarchy_flatten": [0.20, 0.40, 0.60, 0.80, 1.00],
    "relation_delete": [0.20, 0.40, 0.60, 0.80, 1.00],
    "semantic_relabel": [0.10, 0.20, 0.40, 0.60, 0.80],
    "spatial_crop": [0.10, 0.20, 0.40, 0.60, 0.80],
    "building_add": [0.10, 0.20, 0.40, 0.60, 0.80],
    "id_rename": [1.0],
    "object_reorder": [1.0],
    "coordinate_unit": [0.001, 1000.0],
    "cityjson_roundtrip": [1.0],
}

CORE_SWEEPS = {
    "rotation_z": [180.0], "attribute_delete": [0.80], "coordinate_noise": [0.005],
    "object_delete": [0.05, 0.40], "building_delete": [0.10, 0.40],
    "surface_delete": [0.10, 0.40], "hierarchy_flatten": [0.40],
    "semantic_relabel": [0.20], "cityjson_roundtrip": [1.0], "lod2_to_lod1": [1.0],
}


def bit_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left == right).float().mean().item())


def encode(model, graph, relations, device, relation_mode, feature_mode):
    x, edge_index, edge_type = graph_tensors(graph, relations, device, relation_mode, feature_mode,
                                             int(getattr(model, "projection_key", 2026)))
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
    parser.add_argument("--profile", choices=("all", "core"), default="all")
    parser.add_argument("--work-root", help="Directory for temporary attacked CityGML files")
    parser.add_argument("--attack-cache", help="Persistent shared attacked-file cache reused across checkpoints")
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
    if "encoder_type" in checkpoint:
        config = {**config, "encoder_type": checkpoint["encoder_type"]}
    model = create_model(input_dim, config, max(relations.values(), default=0) + 1,
                         int(config["seed"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        clean_bits = {item["id"]: encode(model, clean_graphs[item["id"]], relations, device,
                                                relation_mode, feature_mode)
                      for item in selected}
        negative_scores = [bit_similarity(clean_bits[left["id"]], clean_bits[right["id"]])
                           for left, right in itertools.combinations(selected, 2)]
        rejection_threshold = quantile(negative_scores, 0.95) if negative_scores else None
        curves = {}
        work_root = Path(args.work_root).resolve() if args.work_root else None
        if work_root: work_root.mkdir(parents=True, exist_ok=True)
        attack_cache = Path(args.attack_cache).resolve() if args.attack_cache else None
        if attack_cache: attack_cache.mkdir(parents=True, exist_ok=True)
        context = contextlib.nullcontext(str(attack_cache)) if attack_cache else \
            tempfile.TemporaryDirectory(prefix="cimfusemark_curves_", dir=work_root)
        with context as temporary:
            temporary_root = Path(temporary)
            base_sweeps = CORE_SWEEPS if args.profile == "core" else DEFAULT_SWEEPS
            selected_sweeps = {name: base_sweeps[name] for name in args.attacks} if args.attacks else base_sweeps
            for attack, levels in selected_sweeps.items():
                points = []
                for level in levels:
                    scores, changed, node_ratios, semantic_ratios, coordinate_ratios, valid = [], [], [], [], [], 0
                    for item in selected:
                        source = (DATA_ROOT / item["path"]).resolve()
                        attacked_path = temporary_root / f"{item['id']}__{attack}_{level}.gml"
                        mutation_path = attacked_path.with_suffix(attacked_path.suffix + ".mutation.json")
                        if attacked_path.exists() and mutation_path.exists():
                            mutation = json.loads(mutation_path.read_text(encoding="utf-8"))
                        else:
                            mutation = attack_citygml_xml(source, attacked_path, attack, level,
                                                          seed=int(config["seed"]))
                            if attack_cache:
                                mutation_path.write_text(json.dumps(mutation), encoding="utf-8")
                        try:
                            attacked = build_citygml_graph(attacked_path)
                            bits = encode(model, attacked, relations, device, relation_mode, feature_mode)
                            scores.append(bit_similarity(clean_bits[item["id"]], bits))
                            node_ratios.append(len(attacked.nodes) / max(len(clean_graphs[item["id"]].nodes), 1))
                            clean_semantic = len(clean_graphs[item["id"]].nodes)
                            attacked_semantic = len(attacked.nodes)
                            clean_coordinates = sum(node.point_count for node in clean_graphs[item["id"]].nodes)
                            attacked_coordinates = sum(node.point_count for node in attacked.nodes)
                            semantic_ratios.append(attacked_semantic / max(clean_semantic, 1))
                            coordinate_ratios.append(attacked_coordinates / max(clean_coordinates, 1))
                            valid += 1
                        except ValueError:
                            # A destructively emptied model is an authentication failure, not a missing sample.
                            scores.append(0.0)
                            node_ratios.append(0.0)
                            semantic_ratios.append(0.0)
                            coordinate_ratios.append(0.0)
                        changed.append(int(mutation["changed_elements"]))
                    authentication = ({"auc": auc_from_scores(scores, negative_scores),
                                       **eer_from_scores(scores, negative_scores),
                                       "frr_at_negative_q95": sum(score < rejection_threshold for score in scores) / len(scores)}
                                      if negative_scores else {})
                    points.append({
                        "intensity": level, "models": len(scores),
                        "similarity_mean": statistics.fmean(scores),
                        "similarity_q05": quantile(scores, 0.05),
                        "similarity_minimum": min(scores),
                        "ber_mean": 1.0 - statistics.fmean(scores),
                        "valid_graph_rate": valid / len(scores),
                        "remaining_node_ratio_mean": statistics.fmean(node_ratios),
                        "remaining_node_ratio_minimum": min(node_ratios),
                        "remaining_semantic_node_ratio_mean": statistics.fmean(semantic_ratios),
                        "remaining_coordinate_node_ratio_mean": statistics.fmean(coordinate_ratios),
                        "changed_elements_mean": statistics.fmean(changed),
                        "scores": scores,
                        **authentication,
                    })
                curves[attack] = points

    output = {
        "protocol": {
            "manifest": str(args.manifest), "checkpoint": str(args.checkpoint),
            "split": args.split, "models": len(selected), "device": str(device),
            "relation_mode": relation_mode, "feature_mode": feature_mode,
            "encoder_type": config.get("encoder_type", "rgcn"),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad),
            "fingerprint_bits": int(config["fingerprint_bits"]),
            "negative_pairs": len(negative_scores),
            "negative_mean": statistics.fmean(negative_scores) if negative_scores else None,
            "negative_q95": rejection_threshold,
            "negative_maximum": max(negative_scores) if negative_scores else None,
            "distinct_fingerprints": len({tuple(int(value) for value in bits.tolist())
                                           for bits in clean_bits.values()}),
            "collision_pairs": sum(score == 1.0 for score in negative_scores),
            "negative_scores": negative_scores,
            "clean_fingerprints": {model_id: "".join("1" if int(value) else "0" for value in bits.tolist())
                                   for model_id, bits in clean_bits.items()},
        },
        "curves": curves,
        "warning": "Attack-specific similarity curves; thresholds are not recalibrated at each intensity.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {attack: [{"intensity": point["intensity"],
                         "mean": round(point["similarity_mean"], 6),
                         "q05": round(point["similarity_q05"], 6),
                         "auc": round(point["auc"], 6) if "auc" in point else None,
                         "eer": round(point["eer"], 6) if "eer" in point else None,
                         "frr_at_negative_q95": round(point["frr_at_negative_q95"], 6)
                         if "frr_at_negative_q95" in point else None} for point in points]
               for attack, points in curves.items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
