"""Real XML-level CityGML attacks used before rebuilding the graph."""

from __future__ import annotations

import math
import random
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .citygml_graph import BOUNDARY_TYPES, OBJECT_TYPES
from .core import _local_name


def _coordinate_elements(root: ET.Element) -> list[ET.Element]:
    return [element for element in root.iter() if _local_name(element.tag) in {"pos", "posList"} and element.text]


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _qualified(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def _points_below(element: ET.Element) -> list[tuple[float, float, float]]:
    values = [float(value) for coordinate in _coordinate_elements(element)
              for value in (coordinate.text or "").split()]
    return list(zip(values[0::3], values[1::3], values[2::3]))


def _lod1_box(building: ET.Element, ordinal: int) -> ET.Element | None:
    points = _points_below(building)
    if not points:
        return None
    xs, ys, zs = zip(*points)
    lo, hi = (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
    x0, y0, z0 = lo; x1, y1, z1 = hi
    faces = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(x0,y0,z0)],
        [(x0,y0,z1),(x0,y1,z1),(x1,y1,z1),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y0,z0),(x0,y0,z1),(x1,y0,z1),(x1,y0,z0),(x0,y0,z0)],
        [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0),(x1,y0,z0)],
        [(x1,y1,z0),(x1,y1,z1),(x0,y1,z1),(x0,y1,z0),(x1,y1,z0)],
        [(x0,y1,z0),(x0,y1,z1),(x0,y0,z1),(x0,y0,z0),(x0,y1,z0)],
    ]
    bldg_ns = _namespace(building.tag)
    coordinate = next(iter(_coordinate_elements(building)), None)
    gml_ns = _namespace(coordinate.tag) if coordinate is not None else "http://www.opengis.net/gml/3.2"
    lod = ET.Element(_qualified(bldg_ns, "lod1Solid"))
    solid = ET.SubElement(lod, _qualified(gml_ns, "Solid"),
                          {_qualified(gml_ns, "id"): f"cimfm_lod1_solid_{ordinal}"})
    exterior = ET.SubElement(solid, _qualified(gml_ns, "exterior"))
    shell = ET.SubElement(exterior, _qualified(gml_ns, "Shell"))
    for index, face in enumerate(faces):
        member = ET.SubElement(shell, _qualified(gml_ns, "surfaceMember"))
        polygon = ET.SubElement(member, _qualified(gml_ns, "Polygon"),
                                {_qualified(gml_ns, "id"): f"cimfm_lod1_{ordinal}_{index}"})
        ring_property = ET.SubElement(polygon, _qualified(gml_ns, "exterior"))
        ring = ET.SubElement(ring_property, _qualified(gml_ns, "LinearRing"))
        pos_list = ET.SubElement(ring, _qualified(gml_ns, "posList"), {"srsDimension": "3"})
        pos_list.text = " ".join(f"{value:.12g}" for point in face for value in point)
    return lod


def _unwrap(parent: ET.Element, child: ET.Element) -> None:
    children = list(parent)
    index = children.index(child)
    parent.remove(child)
    for offset, grandchild in enumerate(list(child)):
        parent.insert(index + offset, grandchild)


def attack_citygml_xml(input_path: str | Path, output_path: str | Path, attack: str,
                       severity: float = 0.05, seed: int = 7) -> dict[str, object]:
    """Mutate a complete XML tree and write an independently parseable CityGML file."""
    if attack == "sequential":
        output_path = Path(output_path)
        stages = (("rotation_z", 37.0), ("quantization", max(severity * 0.02, 0.001)),
                  ("attribute_delete", min(severity, 0.8)),
                  ("object_delete", min(severity, 0.7)))
        current = Path(input_path); changed = 0; intermediates = []
        for index, (stage, level) in enumerate(stages):
            target = output_path if index + 1 == len(stages) else output_path.with_suffix(f".stage{index}.gml")
            mutation = attack_citygml_xml(current, target, stage, level, seed + index)
            changed += int(mutation["changed_elements"]); current = target
            if target != output_path: intermediates.append(target)
        for intermediate in intermediates:
            intermediate.unlink(missing_ok=True)
        return {"attack": attack, "severity": severity, "candidate_elements": None,
                "changed_elements": changed, "output": str(output_path)}
    if attack == "cityjson_roundtrip":
        from .cityjson_roundtrip import roundtrip_citygml_cityjson
        output_path = Path(output_path)
        return roundtrip_citygml_cityjson(input_path, output_path, output_path.with_suffix(".city.json"))
    tree = ET.parse(input_path)
    root = tree.getroot()
    rng = random.Random(seed)
    changed = 0
    candidate_count = None
    if attack in {"object_delete", "building_delete", "surface_delete"}:
        parent_map = {child: parent for parent in root.iter() for child in parent}
        semantic = [element for element in root.iter() if _local_name(element.tag) in OBJECT_TYPES]
        top_level = [element for element in semantic if _local_name(element.tag) in {"Building", "Bridge", "Road", "Tunnel"}]
        if attack == "building_delete":
            candidates = top_level[:]
        elif attack == "surface_delete":
            candidates = [element for element in semantic if _local_name(element.tag) in BOUNDARY_TYPES]
        else:
            candidates = [element for element in semantic if element not in top_level]
            if len(top_level) > 1:
                candidates.extend(top_level)
        rng.shuffle(candidates)
        candidate_count = len(candidates)
        count = min(len(candidates), max(1, round(len(candidates) * severity))) if candidates else 0
        if attack == "building_delete" and candidates:
            count = min(count, len(candidates) - 1)
        for element in candidates[:count]:
            parent = parent_map.get(element)
            if parent is not None and element in list(parent):
                parent.remove(element)
                changed += 1
    elif attack == "lod2_to_lod1":
        buildings = [element for element in root.iter() if _local_name(element.tag) == "Building"]
        for ordinal, building in enumerate(buildings):
            lod1 = _lod1_box(building, ordinal)
            if lod1 is None:
                continue
            removable = [child for child in list(building)
                         if _local_name(child.tag).lower().startswith(("lod2", "lod3"))
                         or any(_local_name(item.tag) in BOUNDARY_TYPES for item in child.iter())]
            for child in removable:
                building.remove(child)
            building.append(lod1); changed += 1
    elif attack in {"hierarchy_flatten", "relation_delete"}:
        if attack == "relation_delete":
            for element in root.iter():
                for key in list(element.attrib):
                    if _local_name(key) == "href" and rng.random() < severity:
                        del element.attrib[key]; changed += 1
        parent_map = {child: parent for parent in root.iter() for child in parent}
        boundaries = [element for element in root.iter() if _local_name(element.tag) in BOUNDARY_TYPES]
        candidate_count = len(boundaries)
        rng.shuffle(boundaries)
        count = min(len(boundaries), max(1, round(len(boundaries) * severity))) if boundaries else 0
        for element in boundaries[:count]:
            parent = parent_map.get(element)
            if parent is not None and element in list(parent):
                _unwrap(parent, element); changed += 1
    elif attack == "semantic_relabel":
        mapping = {"WallSurface": "RoofSurface", "RoofSurface": "GroundSurface",
                   "GroundSurface": "WallSurface"}
        candidates = [element for element in root.iter() if _local_name(element.tag) in mapping]
        candidate_count = len(candidates)
        rng.shuffle(candidates)
        count = min(len(candidates), max(1, round(len(candidates) * severity))) if candidates else 0
        for element in candidates[:count]:
            element.tag = _qualified(_namespace(element.tag), mapping[_local_name(element.tag)])
            changed += 1
    elif attack == "id_rename":
        replacements = {}
        for index, element in enumerate(root.iter()):
            for key in list(element.attrib):
                if _local_name(key) == "id":
                    old = element.attrib[key]; new = f"cimfm_{seed}_{index}"
                    element.attrib[key] = new; replacements[old] = new; changed += 1
        for element in root.iter():
            for key, value in list(element.attrib.items()):
                if _local_name(key) == "href" and value.startswith("#") and value[1:] in replacements:
                    element.attrib[key] = "#" + replacements[value[1:]]
    elif attack == "spatial_crop":
        parent_map = {child: parent for parent in root.iter() for child in parent}
        buildings = [element for element in root.iter() if _local_name(element.tag) == "Building"]
        positioned = []
        for element in buildings:
            points = _points_below(element)
            if points:
                positioned.append((sum(point[0] for point in points) / len(points), element))
        positioned.sort(key=lambda item: item[0])
        candidate_count = len(positioned)
        count = min(max(0, len(positioned) - 1), max(1, round(len(positioned) * severity))) if positioned else 0
        for _x, element in positioned[:count]:
            parent = parent_map.get(element)
            if parent is not None and element in list(parent):
                parent.remove(element); changed += 1
    elif attack == "building_add":
        parent_map = {child: parent for parent in root.iter() for child in parent}
        buildings = [element for element in root.iter() if _local_name(element.tag) == "Building"]
        candidate_count = len(buildings)
        count = max(1, round(len(buildings) * severity)) if buildings else 0
        for copy_index, building in enumerate(buildings[:count]):
            member = parent_map.get(building)
            container = parent_map.get(member) if member is not None else None
            if member is None or container is None:
                continue
            duplicate = copy.deepcopy(member)
            for element_index, element in enumerate(duplicate.iter()):
                for key in list(element.attrib):
                    if _local_name(key) == "id":
                        element.attrib[key] = f"{element.attrib[key]}_added_{copy_index}_{element_index}"
                if _local_name(element.tag) in {"pos", "posList"} and element.text:
                    values = [float(value) for value in element.text.split()]
                    shifted = []
                    for x, y, z in zip(values[0::3], values[1::3], values[2::3]):
                        shifted.extend((x + 25.0 * (copy_index + 1), y, z))
                    element.text = " ".join(f"{value:.12g}" for value in shifted)
            container.append(duplicate); changed += 1
    elif attack == "attribute_delete":
        parent_map = {child: parent for parent in root.iter() for child in parent}
        candidates = [
            element for element in root.iter()
            if len(element) == 0 and (element.text or "").strip()
            and _local_name(element.tag) not in {"pos", "posList", "lowerCorner", "upperCorner"}
        ]
        candidate_count = len(candidates)
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
    elif attack in {"translation", "scale", "rotation_z", "quantization", "coordinate_noise"}:
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
                elif attack == "coordinate_noise":
                    sigma = severity * diagonal
                    x, y, z = x + rng.gauss(0, sigma), y + rng.gauss(0, sigma), z + rng.gauss(0, sigma)
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
    return {"attack": attack, "severity": severity,
            "candidate_elements": candidate_count,
            "changed_elements": changed, "output": str(output_path)}
