"""Losses for multi-view robust zero-watermark representation learning."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def multi_positive_nt_xent(embeddings: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised-contrastive NT-Xent where views in the same row are positives.

    Args:
        embeddings: Normalized tensor shaped ``[models, views, dimensions]``.
    """
    models, views, dimensions = embeddings.shape
    flat = F.normalize(embeddings.reshape(models * views, dimensions), dim=1)
    logits = flat @ flat.T / temperature
    identity = torch.eye(models * views, dtype=torch.bool, device=flat.device)
    labels = torch.arange(models, device=flat.device).repeat_interleave(views)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    logits = logits.masked_fill(identity, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) /
             positives.sum(dim=1).clamp_min(1)).mean()


def robust_bit_loss(soft_bits: torch.Tensor, tail_fraction: float = 0.34) -> tuple[torch.Tensor, ...]:
    """Direct bit stability, balance, quantization and worst-view tail losses.

    ``soft_bits`` has shape ``[models, views, bits]`` and view zero is clean.
    """
    clean = soft_bits[:, :1]
    view_mse = (soft_bits[:, 1:] - clean).square().mean(dim=2)
    stability = view_mse.mean()
    balance = soft_bits.mean(dim=(0, 1)).square().mean()
    quantization = (1.0 - soft_bits.abs()).square().mean()
    tail_count = max(1, math.ceil(view_mse.shape[1] * tail_fraction))
    bit_tail = view_mse.topk(tail_count, dim=1).values.mean()
    return stability, balance, quantization, bit_tail


def embedding_tail_loss(embeddings: torch.Tensor, tail_fraction: float = 0.34) -> torch.Tensor:
    clean = embeddings[:, :1]
    distances = 1.0 - F.cosine_similarity(embeddings[:, 1:], clean.expand_as(embeddings[:, 1:]), dim=2)
    tail_count = max(1, math.ceil(distances.shape[1] * tail_fraction))
    return distances.topk(tail_count, dim=1).values.mean()


def bit_separation_loss(clean_soft_bits: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Penalize positive correlation between fingerprints of different models.

    A correlation at or below zero corresponds to an expected Hamming similarity
    no greater than 0.5. Normalization prevents trivially shrinking soft bits.
    """
    normalized = F.normalize(clean_soft_bits, dim=1)
    correlations = normalized @ normalized.T
    mask = ~torch.eye(len(clean_soft_bits), dtype=torch.bool, device=clean_soft_bits.device)
    if not bool(mask.any()):
        return torch.zeros((), device=clean_soft_bits.device)
    return F.relu(correlations[mask] - margin).square().mean()
