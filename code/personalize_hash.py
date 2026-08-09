#!/usr/bin/env python3
"""Personalize a fixed hash projection using only clean registered CIM models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cimfusemark import build_citygml_graph
from cimfusemark.personalization import codebook_similarity, keyed_codebook
from cimfusemark.rgcn import CIMFuseRGCN, graph_tensors, model_digest

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", help="Clean models enrolled in the registration database")
    parser.add_argument("--config", default=str(ROOT / "configs" / "personalize_hash.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "rgcn_personalized.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_config = checkpoint["config"]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = [item for item in manifest["models"] if item.get("split") == args.split]
    if len(selected) < 2:
        raise ValueError("Personalization needs at least two registered CIM models")
    graphs = {item["id"]: build_citygml_graph((DATA_ROOT / item["path"]).resolve()) for item in selected}
    relations = checkpoint["relations"]
    input_dim = len(next(iter(graphs.values())).nodes[0].features)
    model = CIMFuseRGCN(input_dim, int(base_config["hidden_dim"]), int(base_config["embedding_dim"]),
                        max(relations.values(), default=0) + 1, int(base_config["fingerprint_bits"]),
                        int(base_config["seed"])).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    with torch.no_grad():
        embeddings = torch.stack([model.encode(*graph_tensors(graphs[item["id"]], relations, device))
                                  for item in selected])

    registration_seed = int(base_config["seed"]) + int(config["seed_offset"])
    targets = keyed_codebook(len(selected), int(base_config["fingerprint_bits"]),
                             registration_seed).to(device)
    original = model.projection.detach().clone()
    projection = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.AdamW([projection], lr=float(config["learning_rate"]), weight_decay=0.0)
    generator = torch.Generator(device=device); generator.manual_seed(registration_seed)
    last = {}
    for step in range(int(config["steps"])):
        optimizer.zero_grad()
        normalized_projection = F.normalize(projection, dim=1)
        logits = embeddings @ normalized_projection.T
        soft = torch.tanh(logits / float(config["temperature"]))
        code_loss = (soft - targets).square().mean()
        margin_loss = F.relu(float(config["logit_margin"]) - targets * logits).mean()
        anchor_loss = (1.0 - F.cosine_similarity(normalized_projection, original, dim=1)).mean()
        noise = torch.randn(embeddings.shape, generator=generator, device=device) * float(config["embedding_noise_std"])
        noisy_embeddings = F.normalize(embeddings + noise, dim=1)
        noisy_soft = torch.tanh((noisy_embeddings @ normalized_projection.T) /
                                float(config["temperature"]))
        noise_loss = (noisy_soft - soft.detach()).square().mean()
        loss = (float(config["code_weight"]) * code_loss +
                float(config["margin_weight"]) * margin_loss +
                float(config["anchor_weight"]) * anchor_loss +
                float(config["noise_weight"]) * noise_loss)
        loss.backward(); optimizer.step()
        with torch.no_grad(): projection.copy_(F.normalize(projection, dim=1))
        last = {"step": step + 1, "loss": float(loss.detach()), "code": float(code_loss.detach()),
                "margin": float(margin_loss.detach()), "anchor": float(anchor_loss.detach()),
                "noise": float(noise_loss.detach())}
        if step == 0 or step + 1 == int(config["steps"]) or (step + 1) % 200 == 0:
            print(json.dumps(last))

    with torch.no_grad():
        model.projection.copy_(F.normalize(projection, dim=1))
        registered_bits = torch.stack([model.fingerprint(embedding) for embedding in embeddings])
        similarities = codebook_similarity(registered_bits.float() * 2 - 1)
        off_diagonal = similarities[~torch.eye(len(selected), dtype=torch.bool, device=device)]
    registration = {
        "protocol": "clean-only registered-model hash personalization",
        "split": args.split, "registered_ids": [item["id"] for item in selected],
        "registered_ids_digest": hashlib.sha256("\n".join(item["id"] for item in selected).encode()).hexdigest()[:16],
        "config": config, "last_loss": last,
        "registered_negative_mean": float(off_diagonal.mean()),
        "registered_negative_maximum": float(off_diagonal.max()),
        "warning": "Transductive registration result; encoder remains region-disjoint, hash projection sees clean enrolled models only.",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**checkpoint, "state_dict": model.state_dict(), "registration": registration}, output)
    report = output.with_name(output.stem + "_registration.json")
    report.write_text(json.dumps({**registration, "model_digest": model_digest(model)}, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(output), "report": str(report),
                      "registered_negative_mean": registration["registered_negative_mean"],
                      "registered_negative_maximum": registration["registered_negative_maximum"]}))


if __name__ == "__main__":
    main()
