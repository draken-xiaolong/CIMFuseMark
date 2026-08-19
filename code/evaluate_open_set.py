#!/usr/bin/env python3
"""Calibrate a fixed open-set threshold from non-registered validation CIMs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from cimfusemark import build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def empirical_far_threshold(scores: list[float], target_far: float) -> float:
    """Smallest strict cutoff that admits at most floor(FAR*n) calibration scores."""
    descending = sorted(scores, reverse=True)
    allowed = math.floor(target_far * len(descending))
    boundary = descending[allowed] if allowed < len(descending) else descending[-1]
    return math.nextafter(boundary, math.inf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registered-split", default="test")
    parser.add_argument("--calibration-split", default="validation")
    parser.add_argument("--curves", help="Optional robustness JSON to annotate with the fixed threshold")
    parser.add_argument("--output", default=str(ROOT / "results" / "open_set_evaluation.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config, relations = checkpoint["config"], checkpoint["relations"]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    registered = [item for item in manifest["models"] if item.get("split") == args.registered_split]
    calibration = [item for item in manifest["models"] if item.get("split") == args.calibration_split]
    if not registered or not calibration: raise ValueError("Both registered and calibration splits are required")
    first = build_citygml_graph((DATA_ROOT / registered[0]["path"]).resolve())
    model = CIMFuseRGCN(len(first.nodes[0].features), int(config["hidden_dim"]), int(config["embedding_dim"]),
                        max(relations.values(), default=0) + 1, int(config["fingerprint_bits"]),
                        int(config["seed"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    def fingerprint(item):
        graph = build_citygml_graph((DATA_ROOT / item["path"]).resolve())
        x, edge_index, edge_type = graph_tensors(graph, relations, device)
        if checkpoint.get("relation_mode") == "no_edges":
            edge_index, edge_type = edge_index[:, :0], edge_type[:0]
        elif checkpoint.get("relation_mode") == "untyped":
            edge_type = torch.zeros_like(edge_type)
        if checkpoint.get("feature_mode") == "geometry":
            x = x.clone(); x[:, 8:] = 0.0
        return model.fingerprint(model.encode(x, edge_index, edge_type))
    with torch.no_grad():
        registered_bits = {item["id"]: fingerprint(item) for item in registered}
        calibration_bits = {item["id"]: fingerprint(item) for item in calibration}
    impostor_scores = []
    impostor_maxima = {}
    for item in calibration:
        scores = [float((calibration_bits[item["id"]] == value).float().mean())
                  for value in registered_bits.values()]
        impostor_scores.extend(scores); impostor_maxima[item["id"]] = max(scores)
    maximum_scores = list(impostor_maxima.values())
    thresholds = {"far_5pct": empirical_far_threshold(maximum_scores, 0.05),
                  "far_1pct": empirical_far_threshold(maximum_scores, 0.01)}
    pair_thresholds = {"far_5pct": empirical_far_threshold(impostor_scores, 0.05),
                       "far_1pct": empirical_far_threshold(impostor_scores, 0.01)}
    threshold = thresholds["far_5pct"]
    pair_threshold = pair_thresholds["far_5pct"]
    output = {"protocol": {"registered_split": args.registered_split,
                            "calibration_split": args.calibration_split,
                            "registered_models": len(registered), "calibration_models": len(calibration),
                            "threshold_rule": "q95 of each calibration impostor's maximum registered similarity"},
              "open_set": {"threshold": threshold, "pair_threshold": pair_threshold,
                           "thresholds": thresholds, "pair_thresholds": pair_thresholds,
                           "pair_mean": statistics.fmean(impostor_scores),
                           "pair_q95": pair_threshold,
                           "maximum_mean": statistics.fmean(impostor_maxima.values()),
                           "maximum_max": max(impostor_maxima.values()),
                           "observed_open_set_far": sum(score >= threshold for score in impostor_maxima.values()) /
                                                    len(impostor_maxima),
                           "observed_pair_far": sum(score >= pair_threshold for score in impostor_scores) /
                                                len(impostor_scores),
                           "observed_open_set_far_by_target": {
                               name: sum(score >= value for score in maximum_scores) / len(maximum_scores)
                               for name, value in thresholds.items()},
                           "observed_pair_far_by_target": {
                               name: sum(score >= value for score in impostor_scores) / len(impostor_scores)
                               for name, value in pair_thresholds.items()},
                           "impostor_maxima": impostor_maxima}}
    if "registration" in checkpoint:
        output["registration"] = checkpoint["registration"]
    if args.curves:
        curves = json.loads(Path(args.curves).read_text(encoding="utf-8"))["curves"]
        output["fixed_threshold_attacks"] = {
            attack: [{"intensity": point["intensity"],
                      "frr_open_set": sum(score < threshold for score in point["scores"]) / len(point["scores"]),
                      "tar_open_set": sum(score >= threshold for score in point["scores"]) / len(point["scores"]),
                      "tar_at_far_5pct": sum(score >= thresholds["far_5pct"] for score in point["scores"]) /
                                         len(point["scores"]),
                      "tar_at_far_1pct": sum(score >= thresholds["far_1pct"] for score in point["scores"]) /
                                         len(point["scores"]),
                      "frr_pair_threshold": sum(score < pair_threshold for score in point["scores"]) /
                                            len(point["scores"]),
                      "models": len(point["scores"])}
                     for point in points] for attack, points in curves.items()}
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
