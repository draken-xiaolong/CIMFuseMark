#!/usr/bin/env python3
"""Prepare a geographically isolated, Japan-only PLATEAU CityGML corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from prepare_plateau_dataset import DATA_ROOT, download, sha256_file, split_citygml

ROOT = Path(__file__).resolve().parent


def download_archive(source: dict, path: Path) -> None:
    if path.exists() and path.stat().st_size == int(source["file_size"]):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2:
        subprocess.run([
            aria2, "--continue=true", "--max-connection-per-server=8",
            "--split=8", "--min-split-size=16M", "--file-allocation=none",
            f"--dir={path.parent}", f"--out={path.name}", source["url"],
        ], check=True)
    else:
        download(source["url"], path)
    if path.stat().st_size != int(source["file_size"]):
        raise ValueError(f"Archive size mismatch for {source['id']}: {path.stat().st_size}")


def building_entries(archive: zipfile.ZipFile) -> list[str]:
    entries = []
    for info in archive.infolist():
        normalized = info.filename.replace("\\", "/")
        if "/udx/bldg/" not in f"/{normalized}" or not normalized.lower().endswith(".gml"):
            continue
        if "appearance" in normalized.lower() or info.file_size == 0:
            continue
        entries.append((info.file_size, normalized))
    # Large central meshes are substantially more likely to contain explicit LoD2 boundary surfaces.
    return [name for _size, name in sorted(entries, key=lambda item: (-item[0], item[1]))]


def prepare_archive_source(source: dict, config: dict, output_root: Path) -> tuple[list[dict], list[dict]]:
    archive_path = DATA_ROOT / "plateau_multicity_archives" / f"{source['id']}.zip"
    download_archive(source, archive_path)
    cache_root = DATA_ROOT / "plateau_multicity_cache" / source["id"]
    records, selected_files = [], []
    target = int(config["max_rich_buildings_per_archive_source"])
    scan_limit = int(config["max_archive_gml_scans"])
    with zipfile.ZipFile(archive_path) as archive:
        entries = building_entries(archive)
        if not entries:
            raise ValueError(f"No bldg GML entries in {source['id']}")
        for scan_index, entry in enumerate(entries[:scan_limit]):
            if sum(item["building_count"] for item in records) >= target:
                break
            mesh_name = Path(entry).stem.replace("_bldg_6697_op", "")
            extracted = cache_root / f"{mesh_name}.gml"
            if not extracted.exists() or extracted.stat().st_size == 0:
                extracted.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            remaining = target - sum(item["building_count"] for item in records)
            subsource = {**source, "id": f"{source['id']}_{mesh_name}"}
            try:
                rows = split_citygml(
                    subsource, extracted, output_root,
                    int(config["buildings_per_tile"]), remaining,
                    int(config["minimum_boundary_nodes"]),
                )
            except ValueError:
                rows = []
            if rows:
                records.extend(rows)
                selected_files.append({
                    "entry": entry, "path": str(extracted.relative_to(DATA_ROOT)),
                    "sha256": sha256_file(extracted),
                    "bytes": extracted.stat().st_size,
                    "buildings": sum(row["building_count"] for row in rows),
                })
    return records, [{
        **source, "archive_path": str(archive_path.relative_to(DATA_ROOT)),
        "archive_sha256": sha256_file(archive_path), "selected_gml": selected_files,
        "tiles": len(records),
    }]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "plateau_multicity.json"))
    parser.add_argument("--manifest", default=str(DATA_ROOT / "plateau_multicity_manifest.json"))
    parser.add_argument("--output-subdir", default="plateau_multicity_tiles")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest)
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((DATA_ROOT / item["path"]).exists() for item in existing["models"]):
            print(json.dumps(existing["summary"], indent=2)); return

    output_root = DATA_ROOT / args.output_subdir
    models, sources = [], []
    direct_cache = DATA_ROOT / "plateau_multicity_cache" / "direct"
    for source in config["sources"]:
        if source["kind"] == "zip":
            rows, source_rows = prepare_archive_source(source, config, output_root)
        else:
            source_path = direct_cache / f"{source['id']}.gml"
            download(source["url"], source_path)
            rows = split_citygml(
                source, source_path, output_root,
                int(config["buildings_per_tile"]),
                int(config["max_rich_buildings_per_direct_source"]),
                int(config["minimum_boundary_nodes"]),
            )
            source_rows = [{
                **source, "path": str(source_path.relative_to(DATA_ROOT)),
                "sha256": sha256_file(source_path), "tiles": len(rows),
            }]
        if not rows:
            raise ValueError(f"No rich semantic buildings selected from {source['id']}")
        models.extend(rows); sources.extend(source_rows)

    semantic_totals = sum((Counter(item["semantic_counts"]) for item in models), Counter())
    summary = {
        "models": len(models),
        "buildings": sum(item["building_count"] for item in models),
        "boundary_nodes": sum(item["boundary_count"] for item in models),
        "semantic_counts": dict(sorted(semantic_totals.items())),
        "by_split": {split: sum(item["split"] == split for item in models)
                     for split in ("train", "validation", "test")},
        "regions_by_split": {split: sorted({item["region"] for item in models if item["split"] == split})
                             for split in ("train", "validation", "test")},
    }
    manifest = {
        "dataset": config["dataset"], "catalog_api": config["catalog_api"],
        "license_url": config["license_url"], "sources": sources,
        "models": models, "summary": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
