"""Deterministic CityGML -> CityJSON 2.0 -> CityGML round-trip for robustness tests."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .citygml_graph import BOUNDARY_TYPES, GML_ID, GML_ID_OLD
from .core import _local_name


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}" if namespace else name


def _rings(element: ET.Element) -> list[list[tuple[float, float, float]]]:
    rings = []
    for coordinate in element.iter():
        if _local_name(coordinate.tag) not in {"pos", "posList"} or not coordinate.text:
            continue
        values = [float(value) for value in coordinate.text.split()]
        points = list(zip(values[0::3], values[1::3], values[2::3]))
        if len(points) >= 3:
            if points[0] != points[-1]: points.append(points[0])
            rings.append(points)
    return rings


def citygml_to_cityjson(input_path: str | Path) -> tuple[dict, dict]:
    root = ET.parse(input_path).getroot()
    buildings = [element for element in root.iter() if _local_name(element.tag) == "Building"]
    coordinate = next((element for element in root.iter()
                       if _local_name(element.tag) in {"pos", "posList"}), None)
    gml_ns = _namespace(coordinate.tag) if coordinate is not None else "http://www.opengis.net/gml/3.2"
    bldg_ns = _namespace(buildings[0].tag) if buildings else "http://www.opengis.net/citygml/building/2.0"
    member = next((element for element in root.iter() if _local_name(element.tag) == "cityObjectMember"), None)
    core_ns = _namespace(member.tag) if member is not None else _namespace(root.tag)
    vertices, vertex_index, objects = [], {}, {}
    def index(point):
        key = tuple(round(value, 9) for value in point)
        if key not in vertex_index:
            vertex_index[key] = len(vertices); vertices.append(list(key))
        return vertex_index[key]
    for ordinal, building in enumerate(buildings):
        identifier = building.attrib.get(GML_ID) or building.attrib.get(GML_ID_OLD) or f"building_{ordinal}"
        surfaces, values, semantics = [], [], []
        for surface in building.iter():
            surface_type = _local_name(surface.tag)
            if surface_type not in BOUNDARY_TYPES: continue
            semantic_index = len(semantics); semantics.append({"type": surface_type})
            for ring in _rings(surface):
                surfaces.append([[index(point) for point in ring]])
                values.append(semantic_index)
        if not surfaces:
            for ring in _rings(building):
                surfaces.append([[index(point) for point in ring]]); values.append(None)
        attributes = {}
        for child in list(building):
            if len(child) == 0 and (child.text or "").strip():
                attributes[_local_name(child.tag)] = child.text.strip()
        objects[identifier] = {
            "type": "Building", "attributes": attributes,
            "geometry": [{"type": "MultiSurface", "lod": "2", "boundaries": surfaces,
                          "semantics": {"surfaces": semantics, "values": values}}],
        }
    cityjson = {"type": "CityJSON", "version": "2.0", "CityObjects": objects, "vertices": vertices,
                "metadata": {"geographicalExtent": [min((v[0] for v in vertices), default=0),
                                                       min((v[1] for v in vertices), default=0),
                                                       min((v[2] for v in vertices), default=0),
                                                       max((v[0] for v in vertices), default=0),
                                                       max((v[1] for v in vertices), default=0),
                                                       max((v[2] for v in vertices), default=0)]}}
    return cityjson, {"root_tag": root.tag, "root_attrib": root.attrib, "core_ns": core_ns,
                      "bldg_ns": bldg_ns, "gml_ns": gml_ns}


def cityjson_to_citygml(cityjson: dict, context: dict, output_path: str | Path) -> None:
    root = ET.Element(context["root_tag"], dict(context["root_attrib"]))
    vertices = cityjson["vertices"]
    for object_id, city_object in cityjson["CityObjects"].items():
        member = ET.SubElement(root, _q(context["core_ns"], "cityObjectMember"))
        building = ET.SubElement(member, _q(context["bldg_ns"], "Building"),
                                 {_q(context["gml_ns"], "id"): object_id})
        for name, value in city_object.get("attributes", {}).items():
            attribute = ET.SubElement(building, _q(context["core_ns"], name)); attribute.text = str(value)
        for geometry in city_object.get("geometry", []):
            semantic = geometry.get("semantics", {})
            semantic_types = semantic.get("surfaces", [])
            values = semantic.get("values", [])
            for surface_index, surface in enumerate(geometry.get("boundaries", [])):
                value = values[surface_index] if surface_index < len(values) else None
                surface_type = (semantic_types[value].get("type", "ClosureSurface")
                                if isinstance(value, int) and value < len(semantic_types) else "ClosureSurface")
                bounded = ET.SubElement(building, _q(context["bldg_ns"], "boundedBy"))
                semantic_surface = ET.SubElement(bounded, _q(context["bldg_ns"], surface_type))
                multi_property = ET.SubElement(semantic_surface, _q(context["bldg_ns"], "lod2MultiSurface"))
                multi = ET.SubElement(multi_property, _q(context["gml_ns"], "MultiSurface"))
                member_surface = ET.SubElement(multi, _q(context["gml_ns"], "surfaceMember"))
                polygon = ET.SubElement(member_surface, _q(context["gml_ns"], "Polygon"))
                exterior = ET.SubElement(polygon, _q(context["gml_ns"], "exterior"))
                ring = ET.SubElement(exterior, _q(context["gml_ns"], "LinearRing"))
                pos_list = ET.SubElement(ring, _q(context["gml_ns"], "posList"), {"srsDimension": "3"})
                indices = surface[0] if surface else []
                pos_list.text = " ".join(f"{coordinate:.12g}" for index in indices for coordinate in vertices[index])
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def roundtrip_citygml_cityjson(input_path: str | Path, output_gml: str | Path,
                               output_cityjson: str | Path | None = None) -> dict[str, object]:
    cityjson, context = citygml_to_cityjson(input_path)
    if output_cityjson:
        Path(output_cityjson).write_text(json.dumps(cityjson, separators=(",", ":")), encoding="utf-8")
    cityjson_to_citygml(cityjson, context, output_gml)
    return {"attack": "cityjson_roundtrip", "severity": 1.0,
            "candidate_elements": len(cityjson["CityObjects"]),
            "changed_elements": len(cityjson["CityObjects"]), "output": str(output_gml),
            "cityjson": str(output_cityjson) if output_cityjson else None,
            "vertices": len(cityjson["vertices"])}
