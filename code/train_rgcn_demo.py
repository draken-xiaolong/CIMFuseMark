#!/usr/bin/env python3
"""Train and evaluate the first relational-GNN CIM zero-watermark prototype."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from cimfusemark import attack_citygml_xml, build_citygml_graph
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors, model_digest, relation_vocabulary
from run_benchmark import auc_from_scores, eer_from_scores, quantile

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def bit_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left == right).float().mean().item())


def encode(model, graph, relations, device, relation_mode="typed", feature_mode="full"):
    x, edge_index, edge_type = graph_tensors(graph, relations, device)
    if relation_mode == "no_edges":
        edge_index = edge_index[:, :0]
        edge_type = edge_type[:0]
    if feature_mode == "geometry":
        x = x.clone()
        x[:, 8:] = 0.0
    return model.encode(x, edge_index, edge_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "rgcn_demo.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--manifest", default=str(DATA_ROOT / "benchmark_manifest.json"))
    parser.add_argument("--output-prefix", default="rgcn_demo")
    parser.add_argument("--relation-mode", choices=("typed", "untyped", "no_edges"), default="typed")
    parser.add_argument("--feature-mode", choices=("full", "geometry"), default="full")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None: config["epochs"] = args.epochs
    seed = int(config["seed"]); random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    records = []
    with tempfile.TemporaryDirectory(prefix="cimfusemark_rgcn_") as temporary:
        temporary_root = Path(temporary)
        for item in manifest["models"]:
            path = (DATA_ROOT / item["path"]).resolve()
            clean = build_citygml_graph(path)
            views = []
            for attack, severity in (("rotation_z", 0.0), ("quantization", 0.001),
                                     ("attribute_delete", 0.10), ("object_delete", 0.05)):
                attacked_path = temporary_root / f"{item['id']}__{attack}.gml"
                attack_citygml_xml(path, attacked_path, attack, severity, seed=seed)
                views.append((attack, build_citygml_graph(attacked_path)))
            records.append({**item, "clean": clean, "views": views})

        relations = relation_vocabulary([record["clean"] for record in records] +
                                        [graph for record in records for _, graph in record["views"]])
        if args.relation_mode == "untyped":
            relations = {name: 0 for name in relations}
        input_dim = len(records[0]["clean"].nodes[0].features)
        model = CIMFuseRGCN(input_dim, int(config["hidden_dim"]), int(config["embedding_dim"]),
                            max(relations.values(), default=0) + 1,
                            int(config["fingerprint_bits"]), seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        def in_split(record, split_name, family_key):
            return record.get("split") == split_name if "split" in record else record["family"] in config[family_key]
        train = [record for record in records if in_split(record, "train", "train_families")]
        for epoch in range(int(config["epochs"])):
            model.train(); optimizer.zero_grad()
            clean_embeddings, view_embeddings = [], []
            for record in train:
                clean_embedding = encode(model, record["clean"], relations, device,
                                         args.relation_mode, args.feature_mode)
                attack_name, view = record["views"][epoch % len(record["views"])]
                view_embedding = encode(model, view, relations, device,
                                        args.relation_mode, args.feature_mode)
                clean_embeddings.append(clean_embedding); view_embeddings.append(view_embedding)
            clean_stack = torch.stack(clean_embeddings); view_stack = torch.stack(view_embeddings)
            robust = (1.0 - F.cosine_similarity(clean_stack, view_stack)).mean()
            similarities = clean_stack @ clean_stack.T
            family_equal = torch.tensor(
                [[left["family"] == right["family"] for right in train] for left in train],
                dtype=torch.bool, device=device)
            diagonal = torch.eye(len(train), dtype=torch.bool, device=device)
            related_mask = family_equal & ~diagonal
            different_mask = ~family_equal
            related = ((1.0 - similarities[related_mask]).mean()
                       if bool(related_mask.any()) else torch.zeros((), device=device))
            separation = F.relu(similarities[different_mask] - float(config["separation_margin"])).mean()
            soft = torch.stack([model.soft_bits(embedding) for embedding in clean_embeddings + view_embeddings])
            balance = soft.mean(dim=0).square().mean()
            loss = (float(config["robust_weight"]) * (robust + related) +
                    float(config["separation_weight"]) * separation +
                    float(config["balance_weight"]) * balance)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if epoch in {0, int(config["epochs"])-1} or (epoch+1) % 50 == 0:
                print(json.dumps({"epoch": epoch+1, "loss": float(loss.detach()),
                                  "robust": float(robust.detach()), "related": float(related.detach()),
                                  "separation": float(separation.detach()),
                                  "balance": float(balance.detach())}))

        model.eval(); evaluations = {}
        with torch.no_grad():
            for split_name, family_key in (("train", "train_families"),
                                           ("validation", "validation_families"),
                                           ("test", "test_families")):
                selected = [record for record in records if in_split(record, split_name, family_key)]
                clean_bits = {}
                positives = []
                for record in selected:
                    embedding = encode(model, record["clean"], relations, device,
                                       args.relation_mode, args.feature_mode)
                    clean_bits[record["id"]] = model.fingerprint(embedding)
                    for attack, graph in record["views"]:
                        bits = model.fingerprint(encode(model, graph, relations, device,
                                                        args.relation_mode, args.feature_mode))
                        positives.append(bit_similarity(clean_bits[record["id"]], bits))
                negatives = []
                related_scores = []
                for left, right in itertools.combinations(selected, 2):
                    score = bit_similarity(clean_bits[left["id"]], clean_bits[right["id"]])
                    if left["family"] != right["family"]: negatives.append(score)
                    else: related_scores.append(score)
                evaluation = {
                    "models": len(selected), "positive_pairs": len(positives), "negative_pairs": len(negatives),
                    "positive_mean": sum(positives)/len(positives) if positives else None,
                    "positive_q05": quantile(positives, 0.05) if positives else None,
                    "negative_mean": sum(negatives)/len(negatives) if negatives else None,
                    "negative_q95": quantile(negatives, 0.95) if negatives else None,
                    "related_pairs": len(related_scores),
                    "related_mean": sum(related_scores)/len(related_scores) if related_scores else None,
                    "related_minimum": min(related_scores) if related_scores else None,
                }
                if positives and negatives:
                    evaluation.update({"auc": auc_from_scores(positives, negatives), **eer_from_scores(positives, negatives)})
                evaluations[split_name] = evaluation

        output = {
            "config": config, "device": str(device), "relations": relations,
            "relation_mode": args.relation_mode, "feature_mode": args.feature_mode,
            "model_digest": model_digest(model), "splits": evaluations,
            "warning": "Exploratory prototype; thresholds require validation on larger independent regions.",
        }
        result_path = ROOT / "results" / f"{args.output_prefix}_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        torch.save({"state_dict": model.state_dict(), "relations": relations, "config": config,
                    "relation_mode": args.relation_mode, "feature_mode": args.feature_mode},
                   ROOT / "results" / f"{args.output_prefix}.pt")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
