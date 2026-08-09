"""Clean-registration personalization utilities for CIM zero watermarking."""

from __future__ import annotations

import torch


def keyed_codebook(items: int, bits: int, seed: int) -> torch.Tensor:
    """Create deterministic balanced ±1 codewords for registered CIM models."""
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    codes = torch.empty((items, bits), dtype=torch.float32)
    for column in range(bits):
        order = torch.randperm(items, generator=generator)
        codes[:, column] = -1.0
        codes[order[:items // 2], column] = 1.0
        if items % 2:
            codes[order[items // 2], column] = 1.0 if column % 2 else -1.0
    return codes


def codebook_similarity(codebook: torch.Tensor) -> torch.Tensor:
    binary = codebook >= 0
    return (binary[:, None, :] == binary[None, :, :]).float().mean(dim=2)
