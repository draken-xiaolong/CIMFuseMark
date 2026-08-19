#!/usr/bin/env python3
"""Controlled registration-time fine-tuning for CIMFuseMark-Lite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.personalization import codebook_similarity, keyed_codebook
from cimfusemark.rgcn import create_model, graph_tensors, model_digest

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Projection-personalized Lite checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "personalize_encoder_lite.json"))
    parser.add_argument("--registered-split", default="test")
    parser.add_argument("--background-split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); started = time.perf_counter()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_config = {**checkpoint["config"],
                   "encoder_type": checkpoint.get("encoder_type", checkpoint["config"].get("encoder_type", "rgcn"))}
    seed = int(base_config["seed"]) + int(config["seed_offset"])
    random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    registered = [item for item in manifest["models"] if item.get("split") == args.registered_split]
    background = [item for item in manifest["models"] if item.get("split") == args.background_split]
    if len(registered) < 2 or not background:
        raise ValueError("Registered and background splits are required")

    relations = checkpoint["relations"]
    relation_mode = checkpoint.get("relation_mode", "typed")
    feature_mode = checkpoint.get("feature_mode", "geometry_depth")
    attacks = [(str(name), float(level)) for name, level in config["attacks"]]
    with tempfile.TemporaryDirectory(prefix="cimfusemark_lite_personal_") as temporary:
        temporary_root = Path(temporary)
        records = []
        for item_index, item in enumerate(registered):
            source = (DATA_ROOT / item["path"]).resolve()
            clean = build_citygml_graph(source)
            views = []
            for attack_index, (attack, level) in enumerate(attacks):
                target = temporary_root / f"{item['id']}__{attack}_{level}.gml"
                attack_citygml_xml(source, target, attack, level, seed=seed + item_index * 17 + attack_index)
                try: views.append(build_citygml_graph(target))
                except ValueError: views.append(clean)
            records.append((item, clean, views))
        background_graphs = [(item, build_citygml_graph((DATA_ROOT / item["path"]).resolve()))
                             for item in background]
        input_dim = graph_tensors(records[0][1], relations, device, relation_mode,
                                  feature_mode, int(base_config["seed"]))[0].shape[1]
        model = create_model(input_dim, base_config, max(relations.values(), default=0) + 1,
                             int(base_config["seed"])).to(device)
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        registered_tensors = [(graph_tensors(clean, relations, device, relation_mode, feature_mode,
                                             int(base_config["seed"])),
                               [graph_tensors(view, relations, device, relation_mode, feature_mode,
                                              int(base_config["seed"])) for view in views])
                              for _item, clean, views in records]
        background_tensors = [graph_tensors(graph, relations, device, relation_mode, feature_mode,
                                            int(base_config["seed"]))
                              for _item, graph in background_graphs]
        with torch.no_grad():
            teacher = torch.stack([model.encode(*clean) for clean, _views in registered_tensors])
        for parameter in model.parameters(): parameter.requires_grad_(False)
        trainable_modules = (model.conv2, model.pool_score, model.readout)
        for module in trainable_modules:
            for parameter in module.parameters(): parameter.requires_grad_(True)
        # The keyed projection is a buffer in the base model; promote it only during personalization.
        projection_value = model.projection.detach().clone()
        del model._buffers["projection"]
        model.projection = torch.nn.Parameter(projection_value)
        parameter_anchors = {name: value.detach().clone() for name, value in model.named_parameters()
                             if value.requires_grad and name != "projection"}
        encoder_parameters = [value for name, value in model.named_parameters()
                              if value.requires_grad and name != "projection"]
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": float(config["learning_rate"])},
            {"params": [model.projection], "lr": float(config["projection_learning_rate"]),
             "weight_decay": 0.0},
        ], weight_decay=float(config["weight_decay"]))
        targets = keyed_codebook(len(records), int(base_config["fingerprint_bits"]), seed).to(device)
        history = []; model.train()
        for step in range(int(config["steps"])):
            optimizer.zero_grad()
            rstart = (step * int(config["registered_batch_size"])) % len(records)
            registered_indices = [(rstart + offset) % len(records)
                                  for offset in range(int(config["registered_batch_size"]))]
            attack_index = step % len(attacks)
            clean_embeddings = torch.stack([model.encode(*registered_tensors[index][0])
                                             for index in registered_indices])
            attacked_embeddings = torch.stack([model.encode(*registered_tensors[index][1][attack_index])
                                                for index in registered_indices])
            selected_targets = targets[registered_indices]
            clean_soft = torch.tanh(clean_embeddings @ model.projection.T / float(config["temperature"]))
            attacked_soft = torch.tanh(attacked_embeddings @ model.projection.T / float(config["temperature"]))
            code_loss = ((clean_soft-selected_targets).square().mean() +
                         (attacked_soft-selected_targets).square().mean()) / 2
            stability_loss = (clean_soft-attacked_soft).square().mean()
            embedding_anchor = (1-F.cosine_similarity(clean_embeddings, teacher[registered_indices], dim=1)).mean()
            parameter_anchor = torch.stack([
                (value-parameter_anchors[name]).square().mean() for name, value in model.named_parameters()
                if name in parameter_anchors]).mean()
            bstart = (step * int(config["background_batch_size"])) % len(background_tensors)
            background_indices = [(bstart + offset) % len(background_tensors)
                                  for offset in range(int(config["background_batch_size"]))]
            background_embeddings = torch.stack([model.encode(*background_tensors[index])
                                                  for index in background_indices])
            background_soft = torch.tanh(background_embeddings @ model.projection.T /
                                         float(config["temperature"]))
            correlations = background_soft @ targets.T / targets.shape[1]
            background_loss = F.relu(correlations-float(config["background_correlation_margin"])).square().mean()
            loss = (float(config["code_weight"])*code_loss +
                    float(config["stability_weight"])*stability_loss +
                    float(config["embedding_anchor_weight"])*embedding_anchor +
                    float(config["parameter_anchor_weight"])*parameter_anchor +
                    float(config["background_weight"])*background_loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            with torch.no_grad(): model.projection.copy_(F.normalize(model.projection, dim=1))
            row = {"step": step+1, "loss": float(loss.detach()), "code": float(code_loss.detach()),
                   "stability": float(stability_loss.detach()), "embedding_anchor": float(embedding_anchor.detach()),
                   "parameter_anchor": float(parameter_anchor.detach()), "background": float(background_loss.detach()),
                   "attack": attacks[attack_index]}
            if step == 0 or step+1 == int(config["steps"]) or (step+1) % 50 == 0:
                print(json.dumps(row), flush=True); history.append(row)
        model.eval()
        with torch.no_grad():
            enrolled = torch.stack([model.fingerprint(model.encode(*clean)) for clean, _views in registered_tensors])
            similarities = codebook_similarity(enrolled.float()*2-1)
            off_diagonal = similarities[~torch.eye(len(records), dtype=torch.bool, device=device)]
        elapsed = time.perf_counter()-started
        trainable_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
        registration = {
            "protocol": "Lite projection plus partial-encoder personalized fine-tuning",
            "registered_split": args.registered_split, "background_split": args.background_split,
            "registered_models": len(records), "background_models": len(background_tensors),
            "fine_tuned_modules": ["conv2", "pool_score", "readout", "projection"],
            "config": config, "history": history, "elapsed_seconds": elapsed,
            "personalized_trainable_parameters": trainable_count,
            "registered_negative_mean": float(off_diagonal.mean()),
            "registered_negative_maximum": float(off_diagonal.max()),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            "registered_ids_digest": hashlib.sha256("\n".join(item["id"] for item, _c, _v in records).encode()).hexdigest()[:16],
        }
        # Saved tensors load into the usual inference model even though projection was trainable here.
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**checkpoint, "state_dict": model.state_dict(), "registration": registration,
                    "personalization_mode": "partial_encoder_plus_projection"}, output)
        output.with_name(output.stem+"_registration.json").write_text(
            json.dumps({**registration, "model_digest": model_digest(model)}, indent=2), encoding="utf-8")
        print(json.dumps({"checkpoint": str(output), "elapsed_seconds": elapsed,
                          "registered_negative_mean": registration["registered_negative_mean"],
                          "registered_negative_maximum": registration["registered_negative_maximum"]}))


if __name__ == "__main__":
    main()
