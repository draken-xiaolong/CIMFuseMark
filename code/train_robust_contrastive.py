#!/usr/bin/env python3
"""LiteGeoFuseMark-inspired multi-view robust training for CIM zero watermarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import resource
import tempfile
import time
from pathlib import Path

import torch

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import create_model, graph_tensors, model_digest, relation_vocabulary
from cimfusemark.robust_losses import (bit_margin_loss, bit_separation_loss,
                                       embedding_tail_loss, multi_positive_nt_xent,
                                       robust_bit_loss, soft_nc_loss)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def _tensorize(graph, relations, device, relation_mode="typed", feature_mode="full", seed=2026):
    return graph_tensors(graph, relations, device, relation_mode, feature_mode, seed)


def _curriculum_levels(config: dict, attack: str, progress: float) -> list[float]:
    levels = list(config[f"{attack}_levels"])
    if not config.get("curriculum", True):
        fixed = config.get("fixed_attack_levels", {}).get(attack)
        return [float(fixed)] if fixed is not None else [float(levels[-1])]
    boundaries = list(config["curriculum_boundaries"])
    stage = sum(progress >= boundary for boundary in boundaries) + 1
    keep = max(1, round(len(levels) * stage / (len(boundaries) + 1)))
    return levels[:keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "robust_contrastive.json"))
    parser.add_argument("--manifest", default=str(DATA_ROOT / "plateau_manifest.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-prefix", default="rgcn_plateau_robust")
    parser.add_argument("--relation-mode", default="typed")
    parser.add_argument("--feature-mode", default="full")
    parser.add_argument("--encoder-type", choices=("rgcn", "gcn", "graphsage", "gat", "relgat"))
    parser.add_argument("--graph-cache-dir", default=str(ROOT / "results" / "p1_graph_cache"),
                        help="Cache for deterministic preprocessed training graphs; pass an empty value to disable")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.encoder_type is not None:
        config["encoder_type"] = args.encoder_type
    seed = int(config["seed"])
    random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_items = [item for item in manifest["models"] if item.get("split") == "train"]
    configured_families = config.get(
        "training_attack_families",
        ["object_delete", "attribute_delete", "quantization", "rotation"],
    )
    attacks = {family: config[f"{family}_levels"] for family in configured_families}
    if int(config["views_per_model"]) == 0:
        attacks = {}
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    records = []
    with tempfile.TemporaryDirectory(prefix="cimfusemark_robust_") as temporary:
        temporary_root = Path(temporary)
        cache_path = None
        if args.graph_cache_dir:
            signature = json.dumps({"manifest": str(Path(args.manifest).resolve()), "train_items": train_items,
                                    "seed": seed, "attacks": attacks}, sort_keys=True).encode()
            cache_path = Path(args.graph_cache_dir) / f"training_graphs_{hashlib.sha256(signature).hexdigest()[:16]}.pkl"
        if cache_path and cache_path.exists():
            cached = pickle.loads(cache_path.read_bytes())
            records, relations = cached["records"], cached["relations"]
            print(json.dumps({"graph_cache": "hit", "path": str(cache_path), "records": len(records)}))
        else:
            for item in train_items:
                source = (DATA_ROOT / item["path"]).resolve()
                clean = build_citygml_graph(source)
                bank = {}
                for family, levels in attacks.items():
                    xml_attack = "rotation_z" if family == "rotation" else family
                    bank[family] = {}
                    for level in levels:
                        target = temporary_root / f"{item['id']}__{family}_{level}.gml"
                        attack_citygml_xml(source, target, xml_attack, float(level), seed=seed + len(records))
                        try:
                            bank[family][float(level)] = build_citygml_graph(target)
                        except ValueError:
                            pass
                records.append({"id": item["id"], "clean": clean, "bank": bank})
            all_graphs = [record["clean"] for record in records]
            all_graphs += [graph for record in records for family in record["bank"].values() for graph in family.values()]
            relations = relation_vocabulary(all_graphs)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pickle.dumps({"records": records, "relations": relations}, protocol=5))
                print(json.dumps({"graph_cache": "write", "path": str(cache_path), "records": len(records)}))
        for record in records:
            record["clean_tensor"] = _tensorize(record["clean"], relations, device,
                                                  args.relation_mode, args.feature_mode, seed)
            record["bank_tensor"] = {family: {level: _tensorize(graph, relations, device,
                                                                  args.relation_mode, args.feature_mode, seed)
                                               for level, graph in levels.items()}
                                     for family, levels in record["bank"].items()}

        input_dim = records[0]["clean_tensor"][0].shape[1]
        preprocessing_seconds = time.perf_counter() - started
        model = create_model(input_dim, config, len(relations), seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        families = list(attacks)
        history = []
        epochs = int(config["epochs"])
        views_per_model = int(config["views_per_model"])
        training_started = time.perf_counter()
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad()
            progress = epoch / max(epochs - 1, 1)
            forced = ([families[(epoch + offset) % len(families)] for offset in range(views_per_model)]
                      if families else [])
            model_views = []
            for record_index, record in enumerate(records):
                tensors = [record["clean_tensor"]]
                for view_index, family in enumerate(forced):
                    allowed = _curriculum_levels(config, family, progress)
                    available = [float(level) for level in allowed if float(level) in record["bank_tensor"][family]]
                    level = available[(epoch + record_index + view_index) % len(available)]
                    tensors.append(record["bank_tensor"][family][level])
                model_views.append(torch.stack([model.encode(*tensor) for tensor in tensors]))
            embeddings = torch.stack(model_views)
            bit_logits = torch.einsum("bvd,kd->bvk", embeddings, model.projection)
            soft_bits = torch.tanh(bit_logits / float(config["bit_temperature"]))
            contrastive = multi_positive_nt_xent(embeddings, float(config["contrastive_temperature"]))
            stability, balance, quantization, bit_tail = robust_bit_loss(
                soft_bits, float(config["tail_fraction"]))
            embedding_tail = embedding_tail_loss(embeddings, float(config["tail_fraction"]))
            bit_separation = bit_separation_loss(
                soft_bits[:, 0], float(config["bit_separation_margin"]))
            nc_mean, nc_worst = soft_nc_loss(soft_bits)
            bit_margin = bit_margin_loss(bit_logits, float(config.get("bit_margin", 0.0)))
            binary = float(config.get("stability_weight", 1.0)) * stability + \
                     float(config["balance_weight"]) * balance + \
                     float(config["quantization_weight"]) * quantization
            loss = (float(config["contrastive_weight"]) * contrastive +
                    float(config["binary_weight"]) * binary +
                    float(config["embedding_tail_weight"]) * embedding_tail +
                    float(config["bit_tail_weight"]) * bit_tail +
                    float(config["bit_separation_weight"]) * bit_separation +
                    float(config.get("nc_weight", 0.0)) * nc_mean +
                    float(config.get("worst_nc_weight", 0.0)) * nc_worst +
                    float(config.get("bit_margin_weight", 0.0)) * bit_margin)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            row = {"epoch": epoch + 1, "loss": float(loss.detach()), "contrastive": float(contrastive.detach()),
                   "bit_stability": float(stability.detach()), "embedding_tail": float(embedding_tail.detach()),
                   "bit_tail": float(bit_tail.detach()), "bit_separation": float(bit_separation.detach()),
                   "nc_loss": float(nc_mean.detach()), "worst_nc_loss": float(nc_worst.detach()),
                   "bit_margin": float(bit_margin.detach()),
                   "families": forced}
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 50 == 0:
                print(json.dumps(row))
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                history.append(row)

        elapsed = time.perf_counter() - started
        output = {"config": config, "device": str(device), "models": len(records),
                  "relations": relations, "model_digest": model_digest(model), "history": history,
                  "elapsed_seconds": elapsed,
                  "preprocessing_seconds": preprocessing_seconds,
                  "training_seconds": time.perf_counter() - training_started,
                  "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                  "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                  "encoder_type": config.get("encoder_type", "rgcn"),
                  "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                              if parameter.requires_grad),
                  "relation_mode": args.relation_mode, "feature_mode": args.feature_mode,
                  "training_protocol": ("clean plus forced attack-family views; curriculum maximums=" +
                                        json.dumps({name: max(levels) for name, levels in attacks.items()}, sort_keys=True))}
        result_root = ROOT / "results"; result_root.mkdir(parents=True, exist_ok=True)
        (result_root / f"{args.output_prefix}_training.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        torch.save({"state_dict": model.state_dict(), "relations": relations, "config": config,
                    "encoder_type": config.get("encoder_type", "rgcn"),
                    "relation_mode": args.relation_mode, "feature_mode": args.feature_mode,
                    "training_protocol": output["training_protocol"]},
                   result_root / f"{args.output_prefix}.pt")
        print(json.dumps({"checkpoint": str(result_root / f'{args.output_prefix}.pt'),
                          "model_digest": output["model_digest"]}))


if __name__ == "__main__":
    main()
