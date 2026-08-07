"""Real XML-level CityGML attacks used before rebuilding the graph."""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from .citygml_graph import OBJECT_TYPES
from .core import _local_name


def _coordinate_elements(root: ET.Element) -> list[ET.Element]:
    return [element for element in root.iter() if _local_name(element.tag) in {"pos", "posList"} and element.text]


def attack_citygml_xml(input_path: str | Path, output_path: str | Path, attack: str,
                       severity: float = 0.05, seed: int = 7) -> dict[str, object]:
    """Mutate a complete XML tree and write an independently parseable CityGML file."""
    tree = ET.parse(input_path)
    root = tree.getroot()
    rng = random.Random(seed)
    changed = 0
    if attack == "object_delete":
        parent_map = {child: parent for parent in root.iter() for child in parent}
        semantic = [element for element in root.iter() if _local_name(element.tag) in OBJECT_TYPES]
        top_level = [element for element in semantic if _local_name(element.tag) in {"Building", "Bridge", "Road", "Tunnel"}]
        candidates = [element for element in semantic if element not in top_level]
        if len(top_level) > 1:
            candidates.extend(top_level)
        rng.shuffle(candidates)
        count = min(len(candidates), max(1, round(len(candidates) * severity))) if candidates else 0
        for element in candidates[:count]:
            parent = parent_map.get(element)
            if parent is not None and element in list(parent):
                parent.remove(element)
                changed += 1
    elif attack == "attribute_delete":
        parent_map = {child: parent for parent in root.iter() for child in parent}
        candidates = [
            element for element in root.iter()
            if len(element) == 0 and (element.text or "").strip()
            and _local_name(element.tag) not in {"pos", "posList", "lowerCorner", "upperCorner"}
        ]
        rng.shuffle(candidates)
        count = min(len(candidates), max(1, round(len(candidates) * severity))) if candidates else 0
        for element in candidates[:count]:
            parent = parent_map.get(element)
            if parent is not None and element in list(parent):
                parent.remove(element)
                changed += 1
    elif attack == "object_reorder":
        for parent in root.iter():
            children = list(parent)
            if len(children) > 1 and any(any(_local_name(item.tag) in OBJECT_TYPES for item in child.iter()) for child in children):
                shuffled = children[:]
                rng.shuffle(shuffled)
                if shuffled != children:
                    parent[:] = shuffled
                    changed += 1
    elif attack in {"translation", "scale", "rotation_z", "quantization"}:
        coordinates = _coordinate_elements(root)
        all_values = [float(value) for element in coordinates for value in element.text.split()]
        triples = list(zip(all_values[0::3], all_values[1::3], all_values[2::3]))
        if not triples:
            raise ValueError("No explicit coordinate triples found")
        xs, ys, zs = zip(*triples)
        diagonal = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2)
        for element in coordinates:
            values = [float(value) for value in element.text.split()]
            transformed = []
            for x, y, z in zip(values[0::3], values[1::3], values[2::3]):
                if attack == "translation": x, y, z = x + 123.4, y - 56.7, z + 8.9
                elif attack == "scale": x, y, z = x * 3.7, y * 3.7, z * 3.7
                elif attack == "rotation_z":
                    angle = math.radians(severity if severity else 37)
                    c, s = math.cos(angle), math.sin(angle)
                    x, y = c*x-s*y, s*x+c*y
                else:
                    step = max(severity * diagonal, 1e-12)
                    x, y, z = round(x/step)*step, round(y/step)*step, round(z/step)*step
                transformed.extend((x, y, z))
            element.text = " ".join(f"{value:.12g}" for value in transformed)
            changed += 1
    else:
        raise ValueError(f"Unknown XML attack: {attack}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {"attack": attack, "severity": severity, "changed_elements": changed, "output": str(output_path)}
