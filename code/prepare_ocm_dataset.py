#!/usr/bin/env python3
"""Download OCM CityGML and create geographically isolated city-model tiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import urllib.request
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

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
    if path.exists() and zipfile.is_zipfile(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "CIMFuseMark-research/0.4"})
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    if last_error is not None:
        raise RuntimeError(f"Failed to download {url} after 3 attempts") from last_error
    if not zipfile.is_zipfile(temporary):
        raise ValueError(f"Downloaded file is not a ZIP archive: {url}")
    temporary.replace(path)


def _write_tile(root_template: ET.Element, members: list[ET.Element], path: Path) -> None:
    root = ET.Element(root_template.tag, dict(root_template.attrib))
    for member in members:
        root.append(member)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def split_zip(source: dict, archive: Path, output_root: Path,
              buildings_per_tile: int, max_buildings: int) -> list[dict]:
    records: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.lower().endswith((".gml", ".xml"))]
        if not names:
            raise ValueError(f"No GML member in {archive}")
        with bundle.open(names[0]) as stream:
            context = ET.iterparse(stream, events=("start", "end"))
            _event, root_template = next(context)
            batch: list[ET.Element] = []
            total = 0
            tile_index = 0
            for event, element in context:
                if event != "end" or _local_name(element.tag) != "cityObjectMember":
                    continue
                if not any(_local_name(child.tag) == "Building" for child in list(element)):
                    element.clear(); continue
                batch.append(copy.deepcopy(element)); total += 1
                element.clear()
                if len(batch) == buildings_per_tile:
                    tile_id = f"{source['id']}_tile_{tile_index:03d}"
                    tile_path = output_root / source["split"] / f"{tile_id}.gml"
                    _write_tile(root_template, batch, tile_path)
                    records.append({
                        "id": tile_id, "family": tile_id, "region": source["region"],
                        "split": source["split"], "path": str(tile_path.relative_to(DATA_ROOT)),
                        "building_count": len(batch), "source_id": source["id"],
                    })
                    tile_index += 1; batch = []
                if total >= max_buildings:
                    break
            if batch:
                tile_id = f"{source['id']}_tile_{tile_index:03d}"
                tile_path = output_root / source["split"] / f"{tile_id}.gml"
                _write_tile(root_template, batch, tile_path)
                records.append({
                    "id": tile_id, "family": tile_id, "region": source["region"],
                    "split": source["split"], "path": str(tile_path.relative_to(DATA_ROOT)),
                    "building_count": len(batch), "source_id": source["id"],
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "ocm_sources.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cache = DATA_ROOT / "external_cache"
    output_root = DATA_ROOT / "ocm_tiles"
    manifest_path = DATA_ROOT / "ocm_manifest.json"
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((DATA_ROOT / item["path"]).exists() for item in existing["models"]):
            print(json.dumps(existing["summary"], indent=2)); return
    models, sources = [], []
    for source in config["sources"]:
        archive = cache / f"{source['id']}.zip"
        download(source["url"], archive)
        source_records = split_zip(source, archive, output_root,
                                   int(config["buildings_per_tile"]),
                                   int(config["max_buildings_per_source"]))
        models.extend(source_records)
        sources.append({**source, "sha256": sha256_file(archive), "tiles": len(source_records)})
    summary = {
        "models": len(models), "buildings": sum(item["building_count"] for item in models),
        "by_split": {split: sum(item["split"] == split for item in models)
                     for split in ("train", "validation", "test")},
    }
    manifest = {
        "dataset": config["dataset"], "license": config["license"],
        "license_url": config["license_url"],
        "registry_url": config["registry_url"], "sources": sources,
        "models": models, "summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
