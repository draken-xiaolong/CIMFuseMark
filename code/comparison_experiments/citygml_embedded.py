"""Native CityGML embedded-watermark baselines.

The implementations follow the published algorithmic descriptions while using
one shared 64-bit payload and XML coordinate I/O so they can be evaluated on
the same PLATEAU files and attacks as CIMFuseMark.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cimfusemark.core import _local_name


def payload(seed: int = 2026, bits: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, bits, dtype=np.uint8)


def _coordinate_elements(root: ET.Element):
    return [e for e in root.iter() if _local_name(e.tag) in {"pos", "posList"} and e.text]


def _read_tree(path: str | Path):
    tree = ET.parse(path); elements = _coordinate_elements(tree.getroot())
    lengths, points = [], []
    for element in elements:
        values = np.asarray([float(v) for v in element.text.split()], dtype=np.float64)
        usable = len(values) // 3 * 3
        lengths.append((usable, values[usable:]))
        points.append(values[:usable].reshape(-1, 3))
    if not points:
        raise ValueError(f"No coordinates in {path}")
    return tree, elements, lengths, np.concatenate(points)


def _write_tree(tree, elements, lengths, points, target: str | Path):
    cursor = 0
    for element, (usable, tail) in zip(elements, lengths):
        count = usable // 3
        values = points[cursor:cursor + count].reshape(-1); cursor += count
        if len(tail): values = np.concatenate([values, tail])
        element.text = " ".join(f"{v:.12g}" for v in values)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)


def _dct(values: np.ndarray) -> np.ndarray:
    n = len(values)
    k = np.arange(n)[:, None]; x = np.arange(n)[None, :]
    result = np.sqrt(2.0 / n) * (np.cos(np.pi * (x + .5) * k / n) @ values)
    result[0] /= np.sqrt(2.0)
    return result


def _idct(coefficients: np.ndarray) -> np.ndarray:
    n = len(coefficients)
    k = np.arange(n)[:, None]; x = np.arange(n)[None, :]
    adjusted = coefficients.copy(); adjusted[0] /= np.sqrt(2.0)
    return np.sqrt(2.0 / n) * (np.cos(np.pi * (x + .5) * k / n).T @ adjusted)


def _qim(value: float, bit: int, step: float) -> float:
    index = int(np.rint(value / step))
    if index % 2 != int(bit):
        index += 1 if value >= index * step else -1
    return index * step


def _qim_bit(value: float, step: float) -> int:
    return int(np.rint(value / step)) & 1


@dataclass(frozen=True)
class EmbeddedMethod:
    name: str
    citation: str
    fidelity: str
    bits: int = 64

    def embed(self, source, target, watermark): raise NotImplementedError
    def extract(self, path): raise NotImplementedError


class DCTRadialBlind(EmbeddedMethod):
    """Jiang--Kim 2018: QIM of DCT coefficients of sorted spherical radii."""
    def __init__(self, step: float = 0.02):
        super().__init__("jiang18_blind_dct", "Jiang and Kim, IADIS CGVCVIP 2018",
                         "native CityGML paper-guided reproduction")
        object.__setattr__(self, "step", step)

    def embed(self, source, target, watermark):
        tree, elements, lengths, points = _read_tree(source)
        for _ in range(12):
            center = points.mean(0); vectors = points - center
            radii = np.linalg.norm(vectors, axis=1); groups = np.array_split(np.argsort(radii), self.bits)
            for bit, group in zip(watermark, groups):
                if not len(group): continue
                c = _dct(radii[group]); index = int(np.argmax(np.abs(c)))
                if _qim_bit(abs(float(c[index])), self.step) == int(bit): continue
                sign = -1.0 if c[index] < 0 else 1.0
                c[index] = sign * _qim(abs(float(c[index])), int(bit), self.step)
                new_radii = np.maximum(_idct(c), 0.0)
                scale = new_radii / np.maximum(radii[group], 1e-12)
                points[group] = center + vectors[group] * scale[:, None]
        _write_tree(tree, elements, lengths, points, target)

    def extract(self, path):
        _, _, _, points = _read_tree(path); center = points.mean(0)
        radii = np.linalg.norm(points - center, axis=1); groups = np.array_split(np.argsort(radii), self.bits)
        return np.asarray([_qim_bit(abs(float(_dct(radii[g])[np.argmax(np.abs(_dct(radii[g])))])), self.step)
                           if len(g) else 0 for g in groups], dtype=np.uint8)


class OctreeDCTQIM(EmbeddedMethod):
    """Jiang et al. 2019: octant segmentation, radial bins and DCT-QIM."""
    def __init__(self, step: float = 0.02):
        super().__init__("jiang19_octree_dct_qim", "Jiang, Lee and Kim, JPHMT 2019",
                         "native CityGML paper-guided reproduction")
        object.__setattr__(self, "step", step)

    @staticmethod
    def _groups(points, bits):
        center = points.mean(0); vectors = points - center
        octants = ((vectors[:, 0] >= 0).astype(int) * 4 + (vectors[:, 1] >= 0).astype(int) * 2 +
                   (vectors[:, 2] >= 0).astype(int))
        radii = np.linalg.norm(vectors, axis=1); groups = []
        for octant in np.argsort([-np.sum(octants == i) for i in range(8)]):
            idx = np.where(octants == octant)[0]
            if len(idx) < 8: continue
            local = radii[idx]; keep = np.abs(local-local.mean()) <= 1.644 * (local.std() or 1.0)
            idx = idx[keep]
            groups.extend(np.array_split(idx[np.argsort(radii[idx])], max(1, bits // 4)))
            if len(groups) >= bits: break
        return center, vectors, radii, (groups + [np.array([], dtype=int)] * bits)[:bits]

    def embed(self, source, target, watermark):
        tree, elements, lengths, points = _read_tree(source)
        for _ in range(12):
            center, vectors, radii, groups = self._groups(points, self.bits)
            for bit, group in zip(watermark, groups):
                if not len(group): continue
                c = _dct(radii[group]); j = int(np.argmax(np.abs(c)))
                if _qim_bit(abs(float(c[j])), self.step) == int(bit): continue
                sign = -1 if c[j] < 0 else 1; c[j] = sign * _qim(abs(float(c[j])), int(bit), self.step)
                updated = np.maximum(_idct(c), 0); points[group] = center + vectors[group] * (updated / np.maximum(radii[group], 1e-12))[:, None]
        _write_tree(tree, elements, lengths, points, target)

    def extract(self, path):
        _, _, _, points = _read_tree(path); _, _, radii, groups = self._groups(points, self.bits)
        result = []
        for group in groups:
            if not len(group): result.append(0); continue
            c = _dct(radii[group]); result.append(_qim_bit(abs(float(c[np.argmax(np.abs(c))])), self.step))
        return np.asarray(result, dtype=np.uint8)


class GroupCQT(EmbeddedMethod):
    """Jin--Kim 2022: sorted coordinate groups with mean-value QIM."""
    def __init__(self, step: float = 0.05):
        super().__init__("jin22_group_cqt", "Jin and Kim, MTAP 2022",
                         "native CityGML paper-guided reproduction")
        object.__setattr__(self, "step", step)

    def embed(self, source, target, watermark):
        tree, elements, lengths, points = _read_tree(source)
        for _ in range(12):
            axis = int(np.argmax(np.ptp(points, axis=0))); order = np.argsort(points[:, axis])
            for bit, group in zip(watermark, np.array_split(order, self.bits)):
                if not len(group): continue
                mean = float(points[group, axis].mean())
                if _qim_bit(mean, self.step) == int(bit): continue
                points[group, axis] += _qim(mean, int(bit), self.step) - mean
        _write_tree(tree, elements, lengths, points, target)

    def extract(self, path):
        _, _, _, points = _read_tree(path); axis = int(np.argmax(np.ptp(points, axis=0)))
        return np.asarray([_qim_bit(float(points[g, axis].mean()), self.step)
                           for g in np.array_split(np.argsort(points[:, axis]), self.bits)], dtype=np.uint8)


class HomographCityGML(EmbeddedMethod):
    """Hong--Kim 2019: repeated bits in sortable gml:id strings using homoglyphs."""
    roman, greek = "ABEHIKMNOPTXYZ", "ΑΒΕΗІΚΜΝΟΡΤΧΥΖ"
    to_greek = str.maketrans(roman, greek); to_roman = str.maketrans(greek, roman)
    pattern = re.compile(r'(?:[A-Za-z_][\w.-]*:)?id=["\']([^"\']+)["\']')

    def __init__(self):
        super().__init__("hong19_homograph", "Hong and Kim, JPHMT 2019",
                         "native CityGML direct reproduction")

    def embed(self, source, target, watermark):
        text = Path(source).read_text(encoding="utf-8")
        matches = list(self.pattern.finditer(text)); replacements = []
        for rank, match in enumerate(sorted(matches, key=lambda m: (''.join(filter(str.isdigit, m.group(1)))[:3], m.group(1)))):
            value = match.group(1); bit = int(watermark[rank % len(watermark)])
            if bit:
                for i, char in enumerate(value):
                    if char in self.roman:
                        value = value[:i] + char.translate(self.to_greek) + value[i+1:]; break
            replacements.append((match.start(1), match.end(1), value))
        chunks, cursor = [], 0
        for start, end, value in sorted(replacements):
            chunks.extend((text[cursor:start], value)); cursor = end
        chunks.append(text[cursor:]); text = "".join(chunks)
        Path(target).parent.mkdir(parents=True, exist_ok=True); Path(target).write_text(text, encoding="utf-8")

    def extract(self, path):
        text = Path(path).read_text(encoding="utf-8"); matches = list(self.pattern.finditer(text))
        votes = [[] for _ in range(self.bits)]
        for rank, match in enumerate(sorted(matches, key=lambda m: (''.join(filter(str.isdigit, m.group(1)))[:3], m.group(1).translate(self.to_roman)))):
            votes[rank % self.bits].append(any(c in self.greek for c in match.group(1)))
        return np.asarray([int(sum(v) >= len(v)/2) if v else 0 for v in votes], dtype=np.uint8)


def all_embedded_methods():
    return [DCTRadialBlind(), OctreeDCTQIM(), HomographCityGML(), GroupCQT()]
