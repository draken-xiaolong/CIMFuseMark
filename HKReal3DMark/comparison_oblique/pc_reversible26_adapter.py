#!/usr/bin/env python3
"""Batchable adaptation of the official PC-Reversible26 implementation.

The paper's core is retained: normal-angle feature points define local spherical
frames and watermark bits are written to decimal digits of neighbour radii.
This adapter removes fixed file paths and the upstream global-variable bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from hkreal3d.io import load_b3dm_vertices


def unit_normal_estimates(points: np.ndarray, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    k = min(k + 1, len(points))
    ids = NearestNeighbors(n_neighbors=k).fit(points).kneighbors(return_distance=False)
    normals = np.empty_like(points, dtype=np.float64)
    curvature = np.empty(len(points), dtype=np.float64)
    for i, neighbours in enumerate(ids):
        local = points[neighbours[1:]] - points[neighbours[1:]].mean(0)
        values, vectors = np.linalg.eigh(local.T @ local / max(1, len(local)))
        normals[i] = vectors[:, 0]
        curvature[i] = values[0] / max(values.sum(), 1e-12)
    # Normal directions are ambiguous; absolute dot products make the angle stable.
    return normals, curvature


def groups(points: np.ndarray, k: int = 16, feature_fraction: float = .12) -> np.ndarray:
    normals, curvature = unit_normal_estimates(points, k)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(points))).fit(points)
    ids = nn.kneighbors(return_distance=False)
    scores = np.empty(len(points))
    for i, neighbours in enumerate(ids):
        dots = np.clip(np.abs(normals[neighbours[1:]] @ normals[i]), 0, 1)
        scores[i] = np.degrees(np.arccos(dots)).mean() + 180 * curvature[i]
    centered_radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    # Decimal-digit embedding is tiny. Quantization prevents that perturbation
    # from changing feature ordering and therefore the blind grouping itself.
    stable_score = np.round(scores, 1); stable_radius = np.round(centered_radius, 2)
    order = np.lexsort((stable_radius, stable_score))[::-1]
    anchors = order[: max(1, int(len(points) * feature_fraction))]
    used = set(int(x) for x in anchors); result = []
    tree = NearestNeighbors(n_neighbors=min(32, len(points))).fit(points)
    distances, candidates = tree.kneighbors(points[anchors], return_distance=True)
    for anchor, row, distance in zip(anchors, candidates, distances):
        stable = sorted(zip(np.round(distance, 2), row), key=lambda x: (x[0], stable_radius[int(x[1])]))
        neighbours = [int(x) for _, x in stable if int(x) not in used and int(x) != int(anchor)][:2]
        if len(neighbours) == 2:
            result.append((int(anchor), *neighbours)); used.update(neighbours)
    if not result:
        raise RuntimeError("No valid PC-Reversible26 vertex groups")
    return np.asarray(result, dtype=np.int64)


def _digit_embed(radius: np.ndarray, bit: np.ndarray, decimal: int) -> np.ndarray:
    scale = 10.0 ** decimal
    scaled = radius * scale
    return (np.floor(scaled) + (bit + scaled - np.floor(scaled)) / 10.0) / scale


def embed(points: np.ndarray, watermark: np.ndarray, decimal: int = 2) -> tuple[np.ndarray, dict]:
    output = np.asarray(points, dtype=np.float64).copy(); grp = groups(output)
    anchor = np.repeat(grp[:, 0], 2); target = grp[:, 1:].reshape(-1)
    delta = output[target] - output[anchor]; radius = np.linalg.norm(delta, axis=1)
    valid = radius > 1e-12; anchor, target, delta, radius = anchor[valid], target[valid], delta[valid], radius[valid]
    bit_index = (np.floor(radius * 10).astype(np.int64) % len(watermark))
    new_radius = _digit_embed(radius, watermark[bit_index], decimal)
    output[target] = output[anchor] + delta * (new_radius / radius)[:, None]
    return output.astype(np.float32), {"groups": int(len(grp)), "writes": int(len(radius))}


def extract(points: np.ndarray, bits: int, decimal: int = 2) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64); grp = groups(points)
    anchor = np.repeat(grp[:, 0], 2); target = grp[:, 1:].reshape(-1)
    radius = np.linalg.norm(points[target] - points[anchor], axis=1)
    index = np.floor(radius * 10).astype(np.int64) % bits
    value = np.floor(((radius * (10.0 ** decimal)) % 1.0) * 10 + 1e-7).astype(np.int64).clip(0, 1)
    votes = np.zeros((bits, 2), dtype=np.int64)
    for i, v in zip(index, value): votes[i, v] += 1
    recovered = (votes[:, 1] >= votes[:, 0]).astype(np.uint8)
    observed = votes.sum(1) > 0
    return recovered, observed


def main() -> None:
    p = argparse.ArgumentParser(); source=p.add_mutually_exclusive_group(required=True); source.add_argument("--points"); source.add_argument("--b3dm"); p.add_argument("--bits", type=int, default=256); p.add_argument("--seed", type=int, default=2026); p.add_argument("--out")
    a = p.parse_args()
    if a.b3dm:
        points = load_b3dm_vertices(a.b3dm)
    else:
        points = np.load(a.points); points = points["points"] if hasattr(points, "files") else points
        if points.ndim == 3: points = points[0]
    watermark = np.random.default_rng(a.seed).integers(0, 2, a.bits, dtype=np.uint8)
    marked, info = embed(points, watermark); recovered, observed = extract(marked, a.bits)
    result = {**info, "bits": a.bits, "observed_bits": int(observed.sum()), "clean_nc_all": float((recovered == watermark).mean()), "clean_nc_observed": float((recovered[observed] == watermark[observed]).mean()) if observed.any() else 0.0}
    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
