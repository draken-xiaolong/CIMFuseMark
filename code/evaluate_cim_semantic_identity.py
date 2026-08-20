#!/usr/bin/env python3
"""Evaluate CIM identity sensitivity on geometry-identical CityGML variants."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import create_model, lgfm_graph_tensors
from comparison_experiments.baselines import Jiang18, bit_similarity

ROOT = Path(__file__).resolve().parent; DATA_ROOT = ROOT / "data"
VARIANTS = {
    "attribute_removed": ("attribute_delete", 1.0),
    "semantic_relabelled": ("semantic_relabel", 1.0),
    "hierarchy_flattened": ("hierarchy_flatten", 1.0),
    "serialization_reordered_control": ("object_reorder", 1.0),
}


def points(path: Path) -> np.ndarray:
    from comparison_experiments.baselines import citygml_points
    return citygml_points(path)


def same_coordinate_multiset(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape: return False
    order_left = np.lexsort((left[:, 2], left[:, 1], left[:, 0]))
    order_right = np.lexsort((right[:, 2], right[:, 1], right[:, 0]))
    return np.array_equal(left[order_left], right[order_right])


def tensor_bundle(graph, relations, device, seed, geometry_only=False):
    values = list(lgfm_graph_tensors(graph, relations, device, seed))
    if geometry_only:
        values[1] = torch.zeros_like(values[1])       # node semantic branch
        values[3] = torch.zeros((2, 0), dtype=torch.long, device=device)  # semantic/relation edges
        values[4] = torch.zeros(0, dtype=torch.long, device=device)
        values[5] = torch.zeros_like(values[5])       # building-region membership
        values[6] = torch.zeros_like(values[6])       # global semantic evidence
    return tuple(values)


def fingerprint(model, graph, relations, device, seed, geometry_only=False):
    return model.fingerprint(model.encode(*tensor_bundle(graph, relations, device, seed, geometry_only)))


def nc(left, right): return float((left == right).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--work-root")
    args = parser.parse_args(); device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = [item for item in manifest["models"] if item.get("split") == args.split]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config, relations = checkpoint["config"], checkpoint["relations"]
    config = {**config, "encoder_type": checkpoint.get("encoder_type", config.get("encoder_type", "lgfm"))}
    model = create_model(0, config, max(relations.values(), default=0)+1, int(config["seed"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval(); jiang = Jiang18()
    modes = {"cimfusemark_full": False, "cimfusemark_semantic_channels_masked": True}
    scores = {mode: {variant: [] for variant in VARIANTS} for mode in modes}
    scores["jiang18"] = {variant: [] for variant in VARIANTS}
    coordinate_checks, mutations = [], {variant: [] for variant in VARIANTS}
    root = Path(args.work_root).resolve() if args.work_root else None
    if root: root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cim_identity_", dir=root) as temporary, torch.no_grad():
        temporary = Path(temporary)
        for item in items:
            source = (DATA_ROOT / item["path"]).resolve(); clean_graph = build_citygml_graph(source)
            clean_bits = {mode: fingerprint(model, clean_graph, relations, device, int(config["seed"]), masked)
                          for mode, masked in modes.items()}
            clean_jiang = jiang.fingerprint(source); clean_points = points(source)
            for variant, (attack, level) in VARIANTS.items():
                target = temporary / f"{item['id']}__{variant}.gml"
                mutation = attack_citygml_xml(source, target, attack, level, seed=args.seed)
                changed_points = points(target)
                exact = same_coordinate_multiset(clean_points, changed_points)
                coordinate_checks.append(exact); mutations[variant].append(int(mutation["changed_elements"]))
                graph = build_citygml_graph(target)
                for mode, masked in modes.items():
                    scores[mode][variant].append(nc(clean_bits[mode], fingerprint(
                        model, graph, relations, device, int(config["seed"]), masked)))
                scores["jiang18"][variant].append(bit_similarity(clean_jiang, jiang.fingerprint(target)))
    summary = {}
    for method, variants in scores.items():
        summary[method] = {variant: {"mean_nc": statistics.fmean(values),
                                     "mean_identity_separation": 1-statistics.fmean(values),
                                     "minimum_nc": min(values), "scores": values}
                           for variant, values in variants.items()}
    output = {"protocol": {"manifest": args.manifest, "checkpoint": args.checkpoint,
                            "models": len(items), "coordinate_exact_rate": statistics.fmean(coordinate_checks),
                            "note": "Identity changes preserve every coordinate; reorder is an invariance control."},
              "changed_elements_mean": {k: statistics.fmean(v) for k, v in mutations.items()},
              "methods": summary}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({m: {v: round(x["mean_nc"], 4) for v, x in values.items()}
                      for m, values in summary.items()}, indent=2))


if __name__ == "__main__": main()
