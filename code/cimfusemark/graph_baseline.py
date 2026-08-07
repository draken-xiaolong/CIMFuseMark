"""Non-learned multi-relational graph fingerprint baseline."""

from __future__ import annotations

import hashlib
import random
import statistics
from collections import Counter

from .citygml_graph import CIMGraph


def _bucket(text: str, size: int) -> tuple[int, float]:
    digest = hashlib.sha256(text.encode()).digest()
    return digest[0] % size, 1.0 if digest[1] % 2 == 0 else -1.0


def graph_descriptor(graph: CIMGraph) -> list[float]:
    """Permutation-invariant node, relation and feature distribution summary."""
    type_bins = [0.0] * 32
    relation_bins = [0.0] * 24
    for node in graph.nodes:
        index, sign = _bucket(node.node_type, len(type_bins)); type_bins[index] += sign
    for edge in graph.edges:
        index, sign = _bucket(edge.relation, len(relation_bins)); relation_bins[index] += sign
    type_scale = max(sum(abs(value) for value in type_bins), 1.0)
    edge_scale = max(sum(abs(value) for value in relation_bins), 1.0)
    descriptor = [value / type_scale for value in type_bins]
    descriptor.extend(value / edge_scale for value in relation_bins)
    feature_dim = len(graph.nodes[0].features)
    for dimension in range(feature_dim):
        values = [node.features[dimension] for node in graph.nodes]
        descriptor.extend((statistics.fmean(values), statistics.pstdev(values), min(values), max(values)))
    degrees = Counter(edge.source for edge in graph.edges)
    degree_values = [degrees[index] for index in range(len(graph.nodes))]
    descriptor.extend([
        math_log_count(len(graph.nodes)), math_log_count(len(graph.edges)),
        statistics.fmean(degree_values), statistics.pstdev(degree_values),
    ])
    return descriptor


def math_log_count(value: int) -> float:
    import math
    return math.log1p(value) / 10.0


def graph_fingerprint(graph: CIMGraph, bits: int = 256,
                      key: str = "CIMFuseMark-graph-baseline-v1") -> str:
    features = graph_descriptor(graph)
    mean = statistics.fmean(features)
    scale = statistics.pstdev(features) or 1.0
    features = [(value - mean) / scale for value in features]
    output = []
    for bit_index in range(bits):
        seed = int.from_bytes(hashlib.sha256(f"{key}:{bit_index}".encode()).digest()[:8], "big")
        rng = random.Random(seed)
        score = sum(value * rng.gauss(0.0, 1.0) for value in features)
        output.append("1" if score >= 0 else "0")
    return "".join(output)

