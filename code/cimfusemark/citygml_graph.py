"""Multi-relational hierarchical graph construction for CityGML 3.x/2.x."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from .core import Point, _eigenvalues_symmetric_3x3, _local_name

GML_ID = "{http://www.opengis.net/gml/3.2}id"
GML_ID_OLD = "{http://www.opengis.net/gml}id"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

OBJECT_TYPES = {
    "Building", "BuildingPart", "BuildingRoom", "BuildingUnit", "Storey",
    "BuildingInstallation", "WallSurface", "RoofSurface", "GroundSurface",
    "ClosureSurface", "InteriorWallSurface", "FloorSurface", "CeilingSurface",
    "Door", "Window", "Bridge", "BridgePart", "BridgeRoom",
    "BridgeInstallation", "Road", "Railway", "Square", "Track",
    "TrafficSpace", "AuxiliaryTrafficSpace", "WaterBody", "LandUse",
    "ReliefFeature", "Tunnel", "TunnelPart", "CityFurniture", "PlantCover",
    "SolitaryVegetationObject", "CityObjectGroup", "GenericCityObject",
}

BOUNDARY_TYPES = {
    "WallSurface", "RoofSurface", "GroundSurface", "ClosureSurface",
    "InteriorWallSurface", "FloorSurface", "CeilingSurface", "Door", "Window",
}


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    depth: int
    parent: int | None
    point_count: int
    attribute_count: int
    features: list[float]


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    relation: str


@dataclass
class CIMGraph:
    source: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _parse_points(element: ET.Element) -> list[Point]:
    if not element.text:
        return []
    values = [float(value) for value in element.text.split()]
    dimension = int(element.attrib.get("srsDimension", "3"))
    if dimension != 3 or len(values) % 3:
        return []
    return list(zip(values[0::3], values[1::3], values[2::3]))


def _stable_id(element: ET.Element, node_type: str, ordinal: int) -> str:
    explicit = element.attrib.get(GML_ID) or element.attrib.get(GML_ID_OLD)
    if explicit:
        return explicit
    return f"anon:{node_type}:{ordinal}"


def _feature_vector(points: list[Point], center: Point, scale: float, depth: int,
                    attributes: int, node_type: str) -> list[float]:
    if points:
        cx = statistics.fmean(point[0] for point in points)
        cy = statistics.fmean(point[1] for point in points)
        cz = statistics.fmean(point[2] for point in points)
        centered = [(x-cx, y-cy, z-cz) for x, y, z in points]
        covariance = [[statistics.fmean(p[i] * p[j] for p in centered) for j in range(3)] for i in range(3)]
        eigenvalues = [max(0.0, value) for value in _eigenvalues_symmetric_3x3(covariance)]
        eig_sum = sum(eigenvalues) or 1.0
        radii = sorted(math.dist(point, (cx, cy, cz)) / scale for point in points)
        radial_quantiles = [radii[min(len(radii)-1, round(q*(len(radii)-1)))] for q in (0.25, 0.50, 0.90)]
        radial = math.dist((cx, cy, cz), center) / scale
    else:
        eigenvalues, eig_sum, radial_quantiles, radial = [0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0], 0.0
    digest = hashlib.sha256(node_type.encode()).digest()
    type_hash = [0.0] * 8
    type_hash[digest[0] % 8] = 1.0 if digest[1] % 2 == 0 else -1.0
    return [
        math.log1p(len(points)) / 10.0,
        radial,
        *(value / eig_sum for value in eigenvalues),
        *radial_quantiles,
        min(depth, 10) / 10.0,
        math.log1p(attributes) / 5.0,
        1.0 if node_type in BOUNDARY_TYPES else 0.0,
        *type_hash,
    ]


def build_citygml_graph(path: str | Path, spatial_k: int = 3) -> CIMGraph:
    """Build an object hierarchy graph plus XLink and proximity relations."""
    path = Path(path)
    root = ET.parse(path).getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}
    object_elements = [element for element in root.iter() if _local_name(element.tag) in OBJECT_TYPES]
    if not object_elements:
        raise ValueError(f"No supported CityGML objects found in {path}")
    index_of = {element: index for index, element in enumerate(object_elements)}
    id_to_index: dict[str, int] = {}
    node_ids: list[str] = []
    for index, element in enumerate(object_elements):
        node_id = _stable_id(element, _local_name(element.tag), index)
        node_ids.append(node_id)
        id_to_index[node_id] = index

    parents: list[int | None] = []
    depths: list[int] = []
    own_points: list[list[Point]] = [[] for _ in object_elements]
    attribute_counts = [0 for _ in object_elements]
    edges: set[GraphEdge] = set()

    for index, element in enumerate(object_elements):
        cursor = parent_map.get(element)
        semantic_parent = None
        relation_hint = "contains"
        while cursor is not None:
            name = _local_name(cursor.tag)
            if name in {"boundary", "boundedBy"}:
                relation_hint = "bounded_by"
            if cursor in index_of:
                semantic_parent = index_of[cursor]
                break
            cursor = parent_map.get(cursor)
        parents.append(semantic_parent)
        depth = 0 if semantic_parent is None else depths[semantic_parent] + 1
        depths.append(depth)
        if semantic_parent is not None:
            edges.add(GraphEdge(semantic_parent, index, relation_hint))
            edges.add(GraphEdge(index, semantic_parent, "part_of"))

    for element in root.iter():
        name = _local_name(element.tag)
        cursor = parent_map.get(element)
        owner = None
        while cursor is not None:
            if cursor in index_of:
                owner = index_of[cursor]
                break
            cursor = parent_map.get(cursor)
        if owner is None:
            continue
        if name in {"pos", "posList"}:
            own_points[owner].extend(_parse_points(element))
        elif len(element) == 0 and (element.text or "").strip() and name not in {"lowerCorner", "upperCorner"}:
            attribute_counts[owner] += 1

    aggregate_points = [list(points) for points in own_points]
    for index in range(len(object_elements) - 1, -1, -1):
        parent = parents[index]
        if parent is not None:
            aggregate_points[parent].extend(aggregate_points[index])

    all_points = [point for points in own_points for point in points]
    if not all_points:
        raise ValueError(f"No explicit 3D coordinates found in supported objects in {path}")
    center = tuple(statistics.fmean(point[axis] for point in all_points) for axis in range(3))
    scale = math.sqrt(statistics.fmean(math.dist(point, center) ** 2 for point in all_points)) or 1.0

    centroids: list[Point | None] = []
    nodes: list[GraphNode] = []
    for index, element in enumerate(object_elements):
        points = aggregate_points[index]
        centroid = (tuple(statistics.fmean(point[axis] for point in points) for axis in range(3)) if points else None)
        centroids.append(centroid)
        node_type = _local_name(element.tag)
        nodes.append(GraphNode(
            node_id=node_ids[index], node_type=node_type, depth=depths[index], parent=parents[index],
            point_count=len(points), attribute_count=attribute_counts[index],
            features=_feature_vector(points, center, scale, depths[index], attribute_counts[index], node_type),
        ))

    for element in root.iter():
        href = element.attrib.get(XLINK_HREF, "")
        if not href.startswith("#") or href[1:] not in id_to_index:
            continue
        cursor = parent_map.get(element)
        source = None
        while cursor is not None:
            if cursor in index_of:
                source = index_of[cursor]
                break
            cursor = parent_map.get(cursor)
        if source is not None:
            target = id_to_index[href[1:]]
            relation = f"xlink:{_local_name(element.tag)}"
            edges.add(GraphEdge(source, target, relation))
            edges.add(GraphEdge(target, source, "xlink_reverse"))

    valid = [index for index, centroid in enumerate(centroids) if centroid is not None]
    for source in valid:
        candidates = sorted(
            ((math.dist(centroids[source], centroids[target]), target) for target in valid if target != source),
            key=lambda item: (item[0], node_ids[item[1]]),
        )[:max(0, spatial_k)]
        for _distance, target in candidates:
            edges.add(GraphEdge(source, target, "spatial_near"))

    return CIMGraph(
        source=str(path), nodes=nodes,
        edges=sorted(edges, key=lambda edge: (edge.source, edge.target, edge.relation)),
        metadata={
            "schema": "cimfusemark_multirel_graph_v1", "spatial_k": spatial_k,
            "object_types": sorted({node.node_type for node in nodes}),
            "relation_types": sorted({edge.relation for edge in edges}),
            "coordinate_count": len(all_points), "scale": scale,
        },
    )
