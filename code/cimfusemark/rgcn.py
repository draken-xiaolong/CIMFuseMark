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

ENCODER_TYPES = ("rgcn", "gcn", "graphsage", "gat", "relgat")


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


class MeanGraphConv(nn.Module):
    """Small dependency-free GCN/GraphSAGE layer used for fair screening."""
    def __init__(self, in_dim: int, out_dim: int, sage: bool = False):
        super().__init__()
        self.sage = sage
        self.linear = nn.Linear(in_dim * (2 if sage else 1), out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, _edge_type: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            neighbours = torch.zeros_like(x)
        else:
            source, target = edge_index
            neighbours = torch.zeros_like(x); neighbours.index_add_(0, target, x[source])
            degree = torch.zeros(len(x), dtype=x.dtype, device=x.device)
            degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
            neighbours = neighbours / degree.clamp_min(1).unsqueeze(1)
        if self.sage:
            return self.linear(torch.cat([x, neighbours], dim=1))
        return self.linear(x + neighbours)


class GraphAttentionConv(nn.Module):
    """Multi-head GAT with optional relation embeddings in messages and attention."""
    def __init__(self, in_dim: int, out_dim: int, relation_count: int,
                 heads: int = 4, relational: bool = False):
        super().__init__()
        if out_dim % heads:
            raise ValueError("out_dim must be divisible by attention heads")
        self.heads = heads; self.head_dim = out_dim // heads; self.relational = relational
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.att_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_target = nn.Parameter(torch.empty(heads, self.head_dim))
        if relational:
            self.relation = nn.Embedding(relation_count, out_dim)
            self.att_relation = nn.Parameter(torch.empty(heads, self.head_dim))
        else:
            self.relation = None; self.register_parameter("att_relation", None)
        nn.init.xavier_uniform_(self.att_source); nn.init.xavier_uniform_(self.att_target)
        if self.att_relation is not None: nn.init.xavier_uniform_(self.att_relation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        residual = self.self_linear(x)
        if edge_index.numel() == 0:
            return residual
        source, target = edge_index
        projected = self.linear(x).view(len(x), self.heads, self.head_dim)
        messages = projected[source]
        relation = None
        if self.relation is not None:
            relation = self.relation(edge_type).view(-1, self.heads, self.head_dim)
            messages = messages + relation
        logits = (projected[source] * self.att_source).sum(-1) + \
                 (projected[target] * self.att_target).sum(-1)
        if relation is not None:
            logits = logits + (relation * self.att_relation).sum(-1)
        logits = F.leaky_relu(logits, 0.2)
        # Segment softmax without optional scatter dependencies.
        weights = torch.zeros_like(logits)
        for node in torch.unique(target):
            mask = target == node
            weights[mask] = torch.softmax(logits[mask], dim=0)
        aggregated = torch.zeros((len(x), self.heads, self.head_dim), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, target, messages * weights.unsqueeze(-1))
        return residual + aggregated.reshape(len(x), -1)


class CIMFuseRGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 relation_count: int, fingerprint_bits: int, projection_key: int = 2026,
                 encoder_type: str = "rgcn", attention_heads: int = 4):
        super().__init__()
        self.projection_key = projection_key
        self.encoder_type = encoder_type
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        if encoder_type == "rgcn":
            layer = lambda: RelGraphConv(hidden_dim, hidden_dim, relation_count)
        elif encoder_type in {"gcn", "graphsage"}:
            layer = lambda: MeanGraphConv(hidden_dim, hidden_dim, encoder_type == "graphsage")
        elif encoder_type in {"gat", "relgat"}:
            layer = lambda: GraphAttentionConv(hidden_dim, hidden_dim, relation_count,
                                                attention_heads, encoder_type == "relgat")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        self.conv1 = layer(); self.conv2 = layer()
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


def create_model(input_dim: int, config: dict, relation_count: int, seed: int) -> CIMFuseRGCN:
    return CIMFuseRGCN(
        input_dim, int(config["hidden_dim"]), int(config["embedding_dim"]), relation_count,
        int(config["fingerprint_bits"]), seed, config.get("encoder_type", "rgcn"),
        int(config.get("attention_heads", 4)),
    )
