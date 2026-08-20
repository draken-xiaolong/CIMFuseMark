#!/usr/bin/env python3
"""Build one 1024-bit hybrid zero-watermark projection from robust and identity subcodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--personalized", required=True)
    parser.add_argument("--personalized-fraction", required=True, type=float)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.personalized_fraction <= 1.0:
        raise ValueError("personalized-fraction must be in [0, 1]")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    personal = torch.load(args.personalized, map_location="cpu", weights_only=False)
    left = base["state_dict"]["projection"]
    right = personal["state_dict"]["projection"]
    if left.shape != right.shape:
        raise ValueError(f"projection shape mismatch: {left.shape} != {right.shape}")
    generator = torch.Generator(device="cpu"); generator.manual_seed(args.seed)
    count = round(left.shape[0] * args.personalized_fraction)
    selected = torch.randperm(left.shape[0], generator=generator)[:count]
    projection = left.clone(); projection[selected] = right[selected]
    state = dict(base["state_dict"]); state["projection"] = projection
    payload = {**base, "state_dict": state,
               "hybrid_projection": {"personalized_fraction": args.personalized_fraction,
                                     "personalized_bits": count, "selection_seed": args.seed,
                                     "base_checkpoint": str(Path(args.base)),
                                     "personalized_checkpoint": str(Path(args.personalized))}}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print({"output": str(output), "bits": int(left.shape[0]), "personalized_bits": count})


if __name__ == "__main__":
    main()
