"""Minimal dependency-light relational GNN for CIMFuseMark experiments."""

from __future__ import annotations

import hashlib
import random

import torch
from torch import nn
import torch.nn.functional as F

from .citygml_graph import CIMGraph


def relation_vocabulary(graphs: list[CIMGraph]) -> dict[str, int]:
    names = sorted({edge.relation for graph in graphs for edge in graph.edges})
    return {name: index for index, name in enumerate(names)}


FEATURE_GROUPS = {
    "geometry": set(range(8)),
    "geometry_type": set(range(8)) | set(range(11, 19)),
    "geometry_attributes": set(range(8)) | {9},
    "geometry_depth": set(range(8)) | {8},
    "geometry_boundary": set(range(8)) | {10},
    "full": set(range(19)),
}


def _selected_edges(graph: CIMGraph, relation_mode: str, seed: int):
    edges = list(graph.edges)
    if relation_mode == "no_edges":
        return []
    if relation_mode == "hierarchy_only":
        edges = [edge for edge in edges if edge.relation in {"bounded_by", "part_of", "contains"}]
    elif relation_mode == "no_spatial":
        edges = [edge for edge in edges if edge.relation != "spatial_near"]
    elif relation_mode == "no_reverse":
        edges = [edge for edge in edges if edge.relation != "part_of"]
    elif relation_mode == "hierarchy_flattened":
        edges = [edge for edge in edges if edge.relation == "spatial_near"]
    elif relation_mode.startswith("edge_drop_"):
        fraction = int(relation_mode.rsplit("_", 1)[1]) / 100.0
        digest = int.from_bytes(hashlib.sha256(f"{graph.source}:{seed}".encode()).digest()[:8], "big")
        rng = random.Random(digest)
        edges = [edge for edge in edges if rng.random() >= fraction]
    elif relation_mode == "random_rewire":
        digest = int.from_bytes(hashlib.sha256(f"{graph.source}:{seed}:rewire".encode()).digest()[:8], "big")
        rng = random.Random(digest); count = len(graph.nodes)
        edges = [type(edge)(rng.randrange(count), rng.randrange(count), edge.relation) for edge in edges]
    return edges


def graph_tensors(graph: CIMGraph, relations: dict[str, int], device: str | torch.device,
                  relation_mode: str = "typed", feature_mode: str = "full", seed: int = 2026):
    x = torch.tensor([node.features for node in graph.nodes], dtype=torch.float32, device=device)
    if feature_mode not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    keep = FEATURE_GROUPS[feature_mode]
    if len(keep) < x.shape[1]:
        drop = [index for index in range(x.shape[1]) if index not in keep]
        x[:, drop] = 0.0
    edges = _selected_edges(graph, relation_mode, seed)
    if edges:
        edge_index = torch.tensor([[edge.source for edge in edges], [edge.target for edge in edges]],
                                  dtype=torch.long, device=device)
        edge_type = torch.tensor([relations[edge.relation] for edge in edges], dtype=torch.long, device=device)
        if relation_mode == "untyped": edge_type.zero_()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)
    return x, edge_index, edge_type


class RelGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relation_count: int):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.relation_weight = nn.Parameter(torch.empty(relation_count, in_dim, out_dim))
        nn.init.xavier_uniform_(self.relation_weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        output = self.self_linear(x)
        if edge_index.numel() == 0:
            return output
        source, target = edge_index
        messages = torch.bmm(x[source].unsqueeze(1), self.relation_weight[edge_type]).squeeze(1)
        aggregated = torch.zeros_like(output)
        aggregated.index_add_(0, target, messages)
        degree = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        return output + aggregated / degree.clamp_min(1).unsqueeze(1)


class CIMFuseRGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 relation_count: int, fingerprint_bits: int, projection_key: int = 2026):
        super().__init__()
        self.projection_key = projection_key
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.conv1 = RelGraphConv(hidden_dim, hidden_dim, relation_count)
        self.conv2 = RelGraphConv(hidden_dim, hidden_dim, relation_count)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(projection_key)
        projection = torch.randn(fingerprint_bits, embedding_dim, generator=generator)
        self.register_buffer("projection", F.normalize(projection, dim=1), persistent=True)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = F.gelu(self.conv1(h, edge_index, edge_type))
        h = F.gelu(self.conv2(h, edge_index, edge_type))
        mean = h.mean(dim=0)
        maximum = h.max(dim=0).values
        attention = torch.softmax(self.pool_score(h).squeeze(1), dim=0)
        weighted = torch.sum(h * attention.unsqueeze(1), dim=0)
        return F.normalize(self.readout(torch.cat([mean, maximum, weighted])), dim=0)

    def soft_bits(self, embedding: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        return torch.tanh((self.projection @ embedding) / temperature)

    def fingerprint(self, embedding: torch.Tensor) -> torch.Tensor:
        return (self.projection @ embedding >= 0).to(torch.uint8)


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]
