"""Traditional 3D zero-watermark baselines adapted to CityGML.

The direct Jiang18 implementation follows the CityGML paper. Mesh-only methods
are explicitly marked adaptations because CityGML polygons do not provide the
triangular connectivity assumed by their original stable-vertex/SDF pipelines.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from cimfusemark.core import _local_name


def citygml_points(path: str | Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    triples: list[tuple[float, float, float]] = []
    for element in root.iter():
        if _local_name(element.tag) not in {"pos", "posList"} or not element.text:
            continue
        values = [float(value) for value in element.text.split()]
        triples.extend(zip(values[0::3], values[1::3], values[2::3]))
    if not triples:
        raise ValueError(f"No coordinate triples in {path}")
    return np.asarray(triples, dtype=np.float64)


def _normalized(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    scale = float(radii.max()) or 1.0
    return centered / scale, radii / scale


def _histogram_bits(values: np.ndarray, bins: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(bins, dtype=np.uint8)
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return (counts >= counts.mean()).astype(np.uint8)


def _skewness(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    centered = values - values.mean(); deviation = float(np.sqrt(np.mean(centered ** 2)))
    return float(np.mean(centered ** 3) / deviation ** 3) if deviation > 1e-12 else 0.0


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(width, len(values)))
    padded = np.pad(values, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


@dataclass(frozen=True)
class Baseline:
    name: str
    citation: str
    fidelity: str
    bits: int

    def fingerprint(self, path: str | Path) -> np.ndarray:
        raise NotImplementedError

    def timed_fingerprint(self, path: str | Path) -> tuple[np.ndarray, float]:
        start = perf_counter(); bits = self.fingerprint(path)
        return bits, perf_counter() - start


class Jiang18(Baseline):
    def __init__(self, bits: int = 64):
        super().__init__("jiang18_citygml_radial_histogram", "Jiang and Kim, JATIT 2018",
                         "direct CityGML reproduction", bits)

    def fingerprint(self, path: str | Path) -> np.ndarray:
        _, radii = _normalized(citygml_points(path))
        return _histogram_bits(radii, self.bits)


class Lee21(Baseline):
    def __init__(self, bits: int = 64):
        super().__init__("lee21_spherical_skew", "Lee et al., MTAP 2021",
                         "CityGML point-set adaptation", bits)

    def fingerprint(self, path: str | Path) -> np.ndarray:
        points, radii = _normalized(citygml_points(path))
        order = np.argsort(radii); groups = np.array_split(order, self.bits)
        polar = np.arccos(np.clip(points[:, 2] / np.maximum(radii, 1e-12), -1.0, 1.0))
        return np.asarray([_skewness(polar[group]) >= 0.0 for group in groups], dtype=np.uint8)


class Wang19Adapted(Baseline):
    def __init__(self, bins: int = 64):
        super().__init__("wang19_multifeature_adapted", "Wang and Zhan, MTAP 2019",
                         "CityGML adaptation; radial stability proxy replaces OSVETA/SDF", bins * 3)
        self.bins = bins

    def fingerprint(self, path: str | Path) -> np.ndarray:
        points, radii = _normalized(citygml_points(path))
        order = np.argsort(radii); sorted_radii = radii[order]
        radial_change = np.abs(sorted_radii - _moving_average(sorted_radii, 9))
        stable = order[radial_change <= np.median(radial_change)]
        stable_radii = radii[stable]
        chord = np.linalg.norm(np.diff(points[stable], axis=0, append=points[stable[:1]]), axis=1)
        chord /= float(chord.max()) or 1.0
        heights = np.abs(points[stable, 2]); heights /= float(heights.max()) or 1.0
        return np.concatenate([_histogram_bits(stable_radii, self.bins),
                               _histogram_bits(chord, self.bins),
                               _histogram_bits(heights, self.bins)])


class Hu26Adapted(Baseline):
    def __init__(self, bins_per_feature: int = 32, segments: int = 4):
        super().__init__("hu26_radial_fusion_adapted", "Hu et al., Scientific Reports 2026",
                         "CityGML adaptation; multiscale residuals replace mesh EMD", bins_per_feature * 2 * segments)
        self.bins_per_feature = bins_per_feature; self.segments = segments

    def fingerprint(self, path: str | Path) -> np.ndarray:
        _, radii = _normalized(citygml_points(path)); signal = np.sort(radii)
        explicit = np.abs(signal - _moving_average(signal, 9))
        implicit = np.abs(signal - _moving_average(signal, 31))
        explicit /= float(explicit.max()) or 1.0; implicit /= float(implicit.max()) or 1.0
        parts = []
        for indices in np.array_split(np.arange(len(signal)), self.segments):
            parts.extend((_histogram_bits(explicit[indices], self.bins_per_feature),
                          _histogram_bits(implicit[indices], self.bins_per_feature)))
        return np.concatenate(parts)


def all_baselines() -> list[Baseline]:
    return [Jiang18(), Lee21(), Wang19Adapted(), Hu26Adapted()]


def bit_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Fingerprint shapes differ")
    return float(np.mean(left == right))
