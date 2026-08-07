"""Minimal dependency-light relational GNN for CIMFuseMark experiments."""

from __future__ import annotations

import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from .citygml_graph import CIMGraph


def relation_vocabulary(graphs: list[CIMGraph]) -> dict[str, int]:
    names = sorted({edge.relation for graph in graphs for edge in graph.edges})
    return {name: index for index, name in enumerate(names)}


def graph_tensors(graph: CIMGraph, relations: dict[str, int], device: str | torch.device):
    x = torch.tensor([node.features for node in graph.nodes], dtype=torch.float32, device=device)
    if graph.edges:
        edge_index = torch.tensor([[edge.source for edge in graph.edges], [edge.target for edge in graph.edges]],
                                  dtype=torch.long, device=device)
        edge_type = torch.tensor([relations[edge.relation] for edge in graph.edges], dtype=torch.long, device=device)
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

