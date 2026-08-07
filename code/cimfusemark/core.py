"""Small, dependency-free CityGML zero-watermarking baseline."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

Point = tuple[float, float, float]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_citygml(path: str | Path) -> tuple[list[Point], Counter[str]]:
    """Extract explicit 3D coordinate triples and semantic XML tag counts."""
    root = ET.parse(path).getroot()
    points: list[Point] = []
    semantics: Counter[str] = Counter()
    semantic_names = {
        "Building", "BuildingPart", "WallSurface", "RoofSurface",
        "GroundSurface", "ClosureSurface", "Door", "Window",
        "BuildingRoom", "BuildingInstallation", "Storey",
        "Bridge", "BridgePart", "BridgeRoom", "BridgeInstallation",
        "Road", "Railway", "Square", "Track", "TrafficSpace",
        "AuxiliaryTrafficSpace", "WaterBody", "LandUse", "ReliefFeature",
        "Tunnel", "TunnelPart", "CityFurniture", "PlantCover",
        "SolitaryVegetationObject", "CityObjectGroup",
    }

    for element in root.iter():
        name = _local_name(element.tag)
        if name in semantic_names:
            semantics[name] += 1
        if name not in {"pos", "posList"} or not element.text:
            continue
        values = [float(value) for value in element.text.split()]
        dimension = int(element.attrib.get("srsDimension", "3"))
        if dimension != 3 or len(values) % 3:
            continue
        points.extend(zip(values[0::3], values[1::3], values[2::3]))

    if len(points) < 4:
        raise ValueError(f"Expected at least four 3D points in {path}, got {len(points)}")
    return points, semantics


def _eigenvalues_symmetric_3x3(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Stable analytic eigenvalues for a real symmetric 3x3 matrix."""
    a11, a12, a13 = matrix[0]
    _, a22, a23 = matrix[1]
    _, _, a33 = matrix[2]
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    if p1 == 0:
        return sorted([a11, a22, a33], reverse=True)
    q = (a11 + a22 + a33) / 3.0
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2 * p1
    p = math.sqrt(p2 / 6.0)
    b = [[(matrix[i][j] - (q if i == j else 0.0)) / p for j in range(3)] for i in range(3)]
    det_b = (
        b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0])
    )
    phi = math.acos(max(-1.0, min(1.0, det_b / 2.0))) / 3.0
    eig1 = q + 2 * p * math.cos(phi)
    eig3 = q + 2 * p * math.cos(phi + 2 * math.pi / 3)
    eig2 = 3 * q - eig1 - eig3
    return sorted([eig1, eig2, eig3], reverse=True)


def invariant_features(points: Sequence[Point], semantics: Counter[str]) -> list[float]:
    """Return a fixed-length translation/rotation/scale-invariant descriptor."""
    cx = statistics.fmean(p[0] for p in points)
    cy = statistics.fmean(p[1] for p in points)
    cz = statistics.fmean(p[2] for p in points)
    centered = [(x - cx, y - cy, z - cz) for x, y, z in points]
    radii = [math.sqrt(x * x + y * y + z * z) for x, y, z in centered]
    scale = math.sqrt(statistics.fmean(r * r for r in radii)) or 1.0
    normalized = [(x / scale, y / scale, z / scale) for x, y, z in centered]
    radii = sorted(r / scale for r in radii)

    covariance = [[statistics.fmean(p[i] * p[j] for p in normalized) for j in range(3)] for i in range(3)]
    eigenvalues = _eigenvalues_symmetric_3x3(covariance)
    total_eigenvalue = sum(eigenvalues) or 1.0
    features = [value / total_eigenvalue for value in eigenvalues]

    # Radial distribution is invariant to coordinate-system orientation and ordering.
    for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        features.append(radii[min(len(radii) - 1, round(q * (len(radii) - 1)))])
    features.extend([
        statistics.fmean(radii),
        statistics.pstdev(radii),
        statistics.fmean((r - statistics.fmean(radii)) ** 3 for r in radii),
    ])

    semantic_order = [
        "Building", "BuildingPart", "WallSurface", "RoofSurface",
        "GroundSurface", "ClosureSurface", "Door", "Window",
        "BuildingRoom", "BuildingInstallation", "Storey",
        "Bridge", "BridgePart", "BridgeRoom", "BridgeInstallation",
        "Road", "Railway", "Square", "Track", "TrafficSpace",
        "AuxiliaryTrafficSpace", "WaterBody", "LandUse", "ReliefFeature",
        "Tunnel", "TunnelPart", "CityFurniture", "PlantCover",
        "SolitaryVegetationObject", "CityObjectGroup",
    ]
    semantic_total = sum(semantics.values()) or 1
    features.extend(semantics[name] / semantic_total for name in semantic_order)
    return features


def fingerprint(
    points: Sequence[Point],
    semantics: Counter[str],
    bits: int = 256,
    key: str = "CIMFuseMark-demo-v1",
) -> str:
    """Map invariant features to a deterministic key-dependent binary fingerprint."""
    features = invariant_features(points, semantics)
    # Remove the descriptor's common positive offset before angular hashing.
    # Without this step unrelated models produce strongly biased, overly similar bits.
    feature_mean = statistics.fmean(features)
    feature_scale = statistics.pstdev(features) or 1.0
    features = [(feature - feature_mean) / feature_scale for feature in features]
    output: list[str] = []
    for bit_index in range(bits):
        seed_material = f"{key}:{bit_index}".encode()
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(seed)
        projection = sum(feature * rng.gauss(0.0, 1.0) for feature in features)
        output.append("1" if projection >= 0 else "0")
    return "".join(output)


def similarity(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("Fingerprint lengths must match")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def attack_points(points: Sequence[Point], attack: str, severity: float = 0.01, seed: int = 7) -> list[Point]:
    rng = random.Random(seed)
    if attack == "translation":
        return [(x + 123.4, y - 56.7, z + 8.9) for x, y, z in points]
    if attack == "scale":
        return [(x * 3.7, y * 3.7, z * 3.7) for x, y, z in points]
    if attack == "rotation_z":
        angle = math.radians(37)
        c, s = math.cos(angle), math.sin(angle)
        return [(c * x - s * y, s * x + c * y, z) for x, y, z in points]
    if attack == "rotation_3d":
        ax, ay = math.radians(29), math.radians(-41)
        cx, sx, cy, sy = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay)
        return [(cy * x + sy * (sx * y + cx * z), cx * y - sx * z, -sy * x + cy * (sx * y + cx * z)) for x, y, z in points]
    if attack == "noise":
        xs, ys, zs = zip(*points)
        diagonal = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2)
        sigma = severity * diagonal
        return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z + rng.gauss(0, sigma)) for x, y, z in points]
    if attack == "crop":
        kept = list(points)
        rng.shuffle(kept)
        return kept[: max(4, round(len(kept) * (1.0 - severity)))]
    if attack == "spatial_crop":
        # Delete a contiguous slice at the high-x side, closer to real local damage.
        ordered = sorted(points, key=lambda point: point[0])
        return ordered[: max(4, round(len(ordered) * (1.0 - severity)))]
    if attack == "quantization":
        xs, ys, zs = zip(*points)
        diagonal = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2)
        step = max(severity * diagonal, 1e-12)
        return [(round(x / step) * step, round(y / step) * step, round(z / step) * step) for x, y, z in points]
    raise ValueError(f"Unknown attack: {attack}")


def attack_semantics(semantics: Counter[str], severity: float = 0.1, seed: int = 7) -> Counter[str]:
    """Simulate deletion of semantic objects without inventing new categories."""
    rng = random.Random(seed)
    expanded = [name for name, count in semantics.items() for _ in range(count)]
    rng.shuffle(expanded)
    remove_count = min(len(expanded), round(len(expanded) * severity))
    return Counter(expanded[remove_count:])
