#!/usr/bin/env python3
"""Download PLATEAU LoD2 CityGML and create semantic building tiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from cimfusemark.citygml_graph import BOUNDARY_TYPES, OBJECT_TYPES
from cimfusemark.core import _local_name

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size:
        try:
            ET.parse(path)
            return
        except ET.ParseError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CIMFuseMark-research/0.5"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            ET.parse(temporary)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    if last_error is not None:
        raise RuntimeError(f"Failed to download {url} after 3 attempts") from last_error
    temporary.replace(path)


def semantic_counts(member: ET.Element) -> Counter[str]:
    return Counter(_local_name(element.tag) for element in member.iter()
                   if _local_name(element.tag) in OBJECT_TYPES)


def is_rich_building(member: ET.Element, minimum_boundary_nodes: int) -> bool:
    counts = semantic_counts(member)
    return counts["Building"] > 0 and sum(counts[name] for name in BOUNDARY_TYPES) >= minimum_boundary_nodes


def write_tile(root_template: ET.Element, members: list[ET.Element], path: Path) -> None:
    root = ET.Element(root_template.tag, dict(root_template.attrib))
    for member in members:
        root.append(member)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def split_citygml(source: dict, input_path: Path, output_root: Path,
                  buildings_per_tile: int, maximum: int,
                  minimum_boundary_nodes: int) -> list[dict]:
    records: list[dict] = []
    context = ET.iterparse(input_path, events=("start", "end"))
    _event, root_template = next(context)
    batch: list[ET.Element] = []
    selected = 0
    tile_index = 0
    for event, element in context:
        if event != "end" or _local_name(element.tag) != "cityObjectMember":
            continue
        if not is_rich_building(element, minimum_boundary_nodes):
            element.clear()
            continue
        batch.append(copy.deepcopy(element))
        selected += 1
        element.clear()
        if len(batch) == buildings_per_tile or selected == maximum:
            tile_id = f"{source['id']}_tile_{tile_index:03d}"
            tile_path = output_root / source["split"] / f"{tile_id}.gml"
            counts = sum((semantic_counts(member) for member in batch), Counter())
            write_tile(root_template, batch, tile_path)
            records.append({
                "id": tile_id, "family": tile_id, "region": source["region"],
                "split": source["split"], "path": str(tile_path.relative_to(DATA_ROOT)),
                "building_count": counts["Building"],
                "boundary_count": sum(counts[name] for name in BOUNDARY_TYPES),
                "semantic_counts": dict(sorted(counts.items())), "source_id": source["id"],
            })
            tile_index += 1
            batch = []
        if selected >= maximum:
            break
    if batch:
        tile_id = f"{source['id']}_tile_{tile_index:03d}"
        tile_path = output_root / source["split"] / f"{tile_id}.gml"
        counts = sum((semantic_counts(member) for member in batch), Counter())
        write_tile(root_template, batch, tile_path)
        records.append({
            "id": tile_id, "family": tile_id, "region": source["region"],
            "split": source["split"], "path": str(tile_path.relative_to(DATA_ROOT)),
            "building_count": counts["Building"],
            "boundary_count": sum(counts[name] for name in BOUNDARY_TYPES),
            "semantic_counts": dict(sorted(counts.items())), "source_id": source["id"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "plateau_sources.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = DATA_ROOT / "plateau_manifest.json"
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((DATA_ROOT / item["path"]).exists() for item in existing["models"]):
            print(json.dumps(existing["summary"], indent=2)); return

    cache = DATA_ROOT / "plateau_cache"
    output_root = DATA_ROOT / "plateau_tiles"
    models, sources = [], []
    for source in config["sources"]:
        source_path = cache / f"{source['id']}.gml"
        download(source["url"], source_path)
        records = split_citygml(
            source, source_path, output_root, int(config["buildings_per_tile"]),
            int(config["max_rich_buildings_per_source"]), int(config["minimum_boundary_nodes"]),
        )
        if not records:
            raise ValueError(f"No rich semantic buildings selected from {source['id']}")
        models.extend(records)
        sources.append({**source, "sha256": sha256_file(source_path), "tiles": len(records)})

    semantic_totals = sum((Counter(item["semantic_counts"]) for item in models), Counter())
    summary = {
        "models": len(models),
        "buildings": sum(item["building_count"] for item in models),
        "boundary_nodes": sum(item["boundary_count"] for item in models),
        "semantic_counts": dict(sorted(semantic_totals.items())),
        "by_split": {split: sum(item["split"] == split for item in models)
                     for split in ("train", "validation", "test")},
    }
    manifest = {
        "dataset": config["dataset"], "catalog_api": config["catalog_api"],
        "license_url": config["license_url"], "specification": config["specification"],
        "sources": sources, "models": models, "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
