#!/usr/bin/env python3
"""Serial P1 training/evaluation queue with resumable outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT.parent, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "plateau_multicity_manifest.json"))
    parser.add_argument("--matrix", default=str(ROOT / "configs" / "p1_experiments.json"))
    parser.add_argument("--base-config", default=str(ROOT / "configs" / "robust_contrastive.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attack-cache", default="/root/autodl-tmp/CIMFuseMark_p0/p1_attack_cache")
    parser.add_argument("--only-group", choices=("loss", "feature", "graph", "multiseed", "reference"))
    args = parser.parse_args()
    python = sys.executable; results = ROOT / "results"; config_root = results / "p1_configs"
    results.mkdir(exist_ok=True); config_root.mkdir(exist_ok=True)
    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))["experiments"]
    if args.only_group: matrix = [item for item in matrix if item["group"] == args.only_group]

    audit = []
    for item in matrix:
        experiment_id = item["id"]
        relation_mode = item.get("relation_mode", "typed")
        feature_mode = item.get("feature_mode", "full")
        reuse = item.get("reuse")
        prefix = reuse or f"p1_{experiment_id}"
        checkpoint = results / f"{prefix}.pt"
        if not checkpoint.exists():
            config = {**base, **item.get("overrides", {})}
            if "seed" in item: config["seed"] = item["seed"]
            config_path = config_root / f"{experiment_id}.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            run([python, str(ROOT / "train_robust_contrastive.py"), "--config", str(config_path),
                 "--manifest", args.manifest, "--device", args.device, "--output-prefix", prefix,
                 "--relation-mode", relation_mode, "--feature-mode", feature_mode,
                 "--graph-cache-dir", str(results / "p1_graph_cache")])
        curves = results / f"p1_{experiment_id}_core_curves.json"
        opened = results / f"p1_{experiment_id}_open_set.json"
        if not curves.exists():
            run([python, str(ROOT / "evaluate_robustness_curves.py"), "--manifest", args.manifest,
                 "--checkpoint", str(checkpoint), "--split", "test", "--profile", "core",
                 "--device", args.device, "--attack-cache", args.attack_cache, "--output", str(curves)])
        run([python, str(ROOT / "evaluate_open_set.py"), "--manifest", args.manifest,
             "--checkpoint", str(checkpoint), "--registered-split", "test",
             "--calibration-split", "validation", "--curves", str(curves),
             "--device", args.device, "--output", str(opened)])
        audit.append({"id": experiment_id, "group": item["group"], "checkpoint": str(checkpoint),
                      "curves": str(curves), "open_set": str(opened),
                      "relation_mode": relation_mode, "feature_mode": feature_mode})
        (results / "p1_pipeline_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    personalized = results / "p1_efficiency_personalized.pt"
    registration = results / "p1_efficiency_personalized_registration.json"
    if not personalized.exists():
        run([python, str(ROOT / "personalize_hash.py"), "--checkpoint", str(results / "rgcn_multicity_separated.pt"),
             "--manifest", args.manifest, "--split", "test", "--background-split", "validation",
             "--device", args.device, "--output", str(personalized)])
    efficiency = results / "p1_efficiency.json"
    command = [python, str(ROOT / "benchmark_p1_efficiency.py"), "--manifest", args.manifest,
               "--checkpoints", f"rgcn={results / 'rgcn_multicity_separated.pt'}",
               f"deepsets={results / 'deepsets_multicity.pt'}", f"personalized={personalized}",
               "--personalization-report", str(registration), "--device", args.device, "--output", str(efficiency)]
    baselines = results / "multicity_all_baselines.json"
    if baselines.exists(): command.extend(["--baselines", str(baselines)])
    run(command)
    print(json.dumps({"status": "complete", "experiments": len(audit)}))


if __name__ == "__main__":
    main()
