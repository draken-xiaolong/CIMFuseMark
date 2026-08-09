#!/usr/bin/env python3
"""LiteGeoFuseMark-inspired multi-view robust training for CIM zero watermarking."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import torch

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors, model_digest, relation_vocabulary
from cimfusemark.robust_losses import (bit_separation_loss, embedding_tail_loss,
                                       multi_positive_nt_xent, robust_bit_loss)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def _tensorize(graph, relations, device):
    return graph_tensors(graph, relations, device)


def _curriculum_levels(config: dict, attack: str, progress: float) -> list[float]:
    levels = list(config[f"{attack}_levels"])
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
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config["seed"])
    random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    train_items = [item for item in manifest["models"] if item.get("split") == "train"]
    attacks = {
        "object_delete": config["object_delete_levels"],
        "attribute_delete": config["attribute_delete_levels"],
        "quantization": config["quantization_levels"],
        "rotation": config["rotation_levels"],
    }

    records = []
    with tempfile.TemporaryDirectory(prefix="cimfusemark_robust_") as temporary:
        temporary_root = Path(temporary)
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
        for record in records:
            record["clean_tensor"] = _tensorize(record["clean"], relations, device)
            record["bank_tensor"] = {family: {level: _tensorize(graph, relations, device)
                                               for level, graph in levels.items()}
                                     for family, levels in record["bank"].items()}

        input_dim = records[0]["clean_tensor"][0].shape[1]
        model = CIMFuseRGCN(input_dim, int(config["hidden_dim"]), int(config["embedding_dim"]),
                            len(relations), int(config["fingerprint_bits"]), seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        families = list(attacks)
        history = []
        epochs = int(config["epochs"])
        views_per_model = int(config["views_per_model"])
        for epoch in range(epochs):
            model.train(); optimizer.zero_grad()
            progress = epoch / max(epochs - 1, 1)
            forced = [families[(epoch + offset) % len(families)] for offset in range(views_per_model)]
            model_views = []
            for record_index, record in enumerate(records):
                tensors = [record["clean_tensor"]]
                for view_index, family in enumerate(forced):
                    level_key = "rotation" if family == "rotation" else family
                    allowed = _curriculum_levels(config, level_key, progress)
                    available = [float(level) for level in allowed if float(level) in record["bank_tensor"][family]]
                    level = available[(epoch + record_index + view_index) % len(available)]
                    tensors.append(record["bank_tensor"][family][level])
                model_views.append(torch.stack([model.encode(*tensor) for tensor in tensors]))
            embeddings = torch.stack(model_views)
            soft_bits = torch.tanh(torch.einsum("bvd,kd->bvk", embeddings, model.projection) /
                                   float(config["bit_temperature"]))
            contrastive = multi_positive_nt_xent(embeddings, float(config["contrastive_temperature"]))
            stability, balance, quantization, bit_tail = robust_bit_loss(
                soft_bits, float(config["tail_fraction"]))
            embedding_tail = embedding_tail_loss(embeddings, float(config["tail_fraction"]))
            bit_separation = bit_separation_loss(
                soft_bits[:, 0], float(config["bit_separation_margin"]))
            binary = stability + float(config["balance_weight"]) * balance + \
                     float(config["quantization_weight"]) * quantization
            loss = (float(config["contrastive_weight"]) * contrastive +
                    float(config["binary_weight"]) * binary +
                    float(config["embedding_tail_weight"]) * embedding_tail +
                    float(config["bit_tail_weight"]) * bit_tail +
                    float(config["bit_separation_weight"]) * bit_separation)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            row = {"epoch": epoch + 1, "loss": float(loss.detach()), "contrastive": float(contrastive.detach()),
                   "bit_stability": float(stability.detach()), "embedding_tail": float(embedding_tail.detach()),
                   "bit_tail": float(bit_tail.detach()), "bit_separation": float(bit_separation.detach()),
                   "families": forced}
            if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 50 == 0:
                print(json.dumps(row))
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                history.append(row)

        output = {"config": config, "device": str(device), "models": len(records),
                  "relations": relations, "model_digest": model_digest(model), "history": history,
                  "training_protocol": "clean plus three forced attack-family views; curriculum up to 60% object deletion"}
        result_root = ROOT / "results"; result_root.mkdir(parents=True, exist_ok=True)
        (result_root / f"{args.output_prefix}_training.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        torch.save({"state_dict": model.state_dict(), "relations": relations, "config": config,
                    "relation_mode": "typed", "feature_mode": "full", "training_protocol": output["training_protocol"]},
                   result_root / f"{args.output_prefix}.pt")
        print(json.dumps({"checkpoint": str(result_root / f'{args.output_prefix}.pt'),
                          "model_digest": output["model_digest"]}))


if __name__ == "__main__":
    main()
