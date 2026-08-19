#!/usr/bin/env python3
"""Separate XML parsing, graph construction, tensorization and inference costs."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from cimfusemark import build_citygml_graph
from cimfusemark.rgcn import create_model, graph_tensors

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def quantile(values, q):
    ordered = sorted(values); position = q * (len(ordered)-1); lower = int(position); upper = min(lower+1, len(ordered)-1)
    return ordered[lower] * (upper-position) + ordered[upper] * (position-lower)


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def load_model(path: Path, first_graph, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False); config = checkpoint["config"]
    relations = checkpoint["relations"]
    input_dim = graph_tensors(first_graph, relations, device, checkpoint.get("relation_mode", "typed"),
                              checkpoint.get("feature_mode", "full"), int(config["seed"]))[0].shape[1]
    config = {**config, "encoder_type": checkpoint.get("encoder_type", config.get("encoder_type", "rgcn"))}
    model = create_model(input_dim, config, max(relations.values(), default=0)+1, int(config["seed"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return checkpoint, model, relations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path entries")
    parser.add_argument("--baselines")
    parser.add_argument("--personalization-report")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = [item for item in manifest["models"] if item.get("split") == "test"]
    paths = [(DATA_ROOT / item["path"]).resolve() for item in items]
    parse_times, build_times, graphs = [], [], []
    for path in paths:
        started = time.perf_counter(); ET.parse(path); parse_times.append(time.perf_counter()-started)
        started = time.perf_counter(); graphs.append(build_citygml_graph(path)); build_times.append(time.perf_counter()-started)
    report = {"protocol": {"models": len(items), "device": str(device), "single_model_repeats": 3},
              "xml_parse_ms": {"mean": statistics.fmean(parse_times)*1000, "median": statistics.median(parse_times)*1000,
                               "q95": quantile(parse_times, .95)*1000},
              "graph_build_including_parse_ms": {"mean": statistics.fmean(build_times)*1000,
                                                   "median": statistics.median(build_times)*1000,
                                                   "q95": quantile(build_times, .95)*1000},
              "graph_construction_estimate_ms": {"mean": statistics.fmean(max(0,b-p) for b,p in zip(build_times,parse_times))*1000},
              "models": {}, "scaling": []}
    sizes = [(len(graph.nodes), len(graph.edges), graph) for graph in graphs]
    for entry in args.checkpoints:
        name, raw_path = entry.split("=", 1); path = Path(raw_path)
        checkpoint, model, relations = load_model(path, graphs[0], device)
        relation_mode = checkpoint.get("relation_mode", "typed"); feature_mode = checkpoint.get("feature_mode", "full")
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        tensor_times, inference_times, tensor_bytes = [], [], []
        with torch.no_grad():
            for graph in graphs:
                started = time.perf_counter()
                tensors = graph_tensors(graph, relations, device, relation_mode, feature_mode,
                                        int(checkpoint["config"]["seed"]))
                synchronize(device); tensor_times.append(time.perf_counter()-started)
                tensor_bytes.append(sum(t.numel()*t.element_size() for t in tensors))
                samples = []
                for _ in range(3):
                    started = time.perf_counter(); model.fingerprint(model.encode(*tensors)); synchronize(device)
                    samples.append(time.perf_counter()-started)
                inference_times.append(statistics.median(samples))
        parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        report["models"][name] = {
            "trainable_parameters": parameters, "checkpoint_bytes": path.stat().st_size,
            "tensorize_ms_mean": statistics.fmean(tensor_times)*1000,
            "inference_ms_mean": statistics.fmean(inference_times)*1000,
            "inference_ms_median": statistics.median(inference_times)*1000,
            "batch_64_seconds": sum(inference_times), "models_per_second": len(items)/sum(inference_times),
            "tensor_bytes_mean": statistics.fmean(tensor_bytes),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "relation_mode": relation_mode, "feature_mode": feature_mode,
        }
        if name == "rgcn":
            for nodes, edges, graph in sizes:
                tensor = graph_tensors(graph, relations, device, relation_mode, feature_mode,
                                       int(checkpoint["config"]["seed"])); started = time.perf_counter()
                with torch.no_grad(): model.fingerprint(model.encode(*tensor))
                synchronize(device)
                report["scaling"].append({"nodes": nodes, "edges": edges,
                                           "inference_ms": (time.perf_counter()-started)*1000,
                                           "tensor_bytes": sum(t.numel()*t.element_size() for t in tensor)})
    if args.baselines:
        baseline = json.loads(Path(args.baselines).read_text(encoding="utf-8"))
        report["traditional_runtime_ms"] = {name: value["runtime_ms_mean"] for name, value in baseline["methods"].items()}
    training_reports = []
    for path in (ROOT / "results").glob("p1_*_training.json"):
        row = json.loads(path.read_text(encoding="utf-8")); training_reports.append({
            "name": path.stem, "elapsed_seconds": row.get("elapsed_seconds"),
            "preprocessing_seconds": row.get("preprocessing_seconds"), "training_seconds": row.get("training_seconds"),
            "peak_gpu_memory_bytes": row.get("peak_gpu_memory_bytes"), "peak_process_rss_bytes": row.get("peak_process_rss_bytes")})
    report["training_runs"] = training_reports
    if args.personalization_report and Path(args.personalization_report).exists():
        registration = json.loads(Path(args.personalization_report).read_text(encoding="utf-8"))
        report["personalization"] = {key: registration.get(key) for key in
                                     ("elapsed_seconds", "peak_gpu_memory_bytes", "peak_process_rss_bytes",
                                      "registered_negative_mean", "registered_negative_maximum")}
    report["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "models": list(report["models"]), "scaling_points": len(report["scaling"])}))


if __name__ == "__main__":
    main()
