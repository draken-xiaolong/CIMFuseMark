from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import trimesh


def load_b3dm_mesh(path: str | Path) -> trimesh.Trimesh:
    """Read and concatenate transformed triangle primitives from a b3dm tile."""
    blob = Path(path).read_bytes()
    if len(blob) < 28:
        raise ValueError(f"Truncated b3dm: {path}")
    magic, version, byte_length, ft_json, ft_bin, bt_json, bt_bin = struct.unpack(
        "<4s6I", blob[:28]
    )
    if magic != b"b3dm" or version != 1 or byte_length > len(blob):
        raise ValueError(f"Unsupported b3dm header: {path}")
    offset = 28 + ft_json + ft_bin + bt_json + bt_bin
    scene = trimesh.load(io.BytesIO(blob[offset:byte_length]), file_type="glb", force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node)
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        if len(mesh.vertices) and len(mesh.faces):
            meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No triangle mesh geometry in {path}")
    return trimesh.util.concatenate(meshes)


def load_b3dm_vertices(path: str | Path) -> np.ndarray:
    """Read all transformed mesh vertices from a legacy Cesium b3dm container."""
    return np.asarray(load_b3dm_mesh(path).vertices, dtype=np.float32)


def normalized_sample(vertices: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic point sample with translation, scale and PCA-frame normalization."""
    rng = np.random.default_rng(seed)
    if len(vertices) >= count:
        idx = rng.choice(len(vertices), count, replace=False)
    else:
        idx = rng.choice(len(vertices), count, replace=True)
    points = vertices[idx].astype(np.float64)
    points -= np.median(points, axis=0, keepdims=True)
    scale = np.quantile(np.linalg.norm(points, axis=1), 0.95)
    points /= max(float(scale), 1e-8)
    covariance = points.T @ points / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    vectors = vectors[:, np.argsort(values)[::-1]]
    points = points @ vectors
    # Resolve PCA sign ambiguity with third central moments.
    signs = np.sign(np.mean(points ** 3, axis=0)); signs[signs == 0] = 1
    points *= signs
    return points.astype(np.float32)
