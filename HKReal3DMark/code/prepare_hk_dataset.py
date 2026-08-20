#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hkreal3d.io import load_b3dm_vertices, normalized_sample


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/Volumes/SANDISK-ELE/HKReal3DMarkData/raw/tiles_f2")
    ap.add_argument("--out", default="/Volumes/SANDISK-ELE/HKReal3DMarkData/converted/hk_points")
    ap.add_argument("--per-region", type=int, default=24)
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--max-files", type=int, default=360)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in raw.rglob("*.b3dm"):
        if path.name.startswith("._") or path.stat().st_size < 4096:
            continue
        rel = path.relative_to(raw)
        # Prefer detailed leaf tiles. The API uses L1 for the finest visible payloads.
        if "_L1" in path.stem:
            groups[rel.parts[0]].append(path)
    selected = []
    for region in sorted(groups, key=lambda x: int(x) if x.isdigit() else x):
        paths = sorted(groups[region], key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
        selected.extend(paths[:args.per_region])
    selected = selected[:args.max_files]
    rows, failures = [], []
    arrays = []
    for index, path in enumerate(selected):
        rel = str(path.relative_to(raw)); seed = stable_seed(rel)
        try:
            vertices = load_b3dm_vertices(path)
            points = normalized_sample(vertices, args.points, seed)
            arrays.append(points)
            rows.append({"id": f"hk_{index:04d}", "region": path.relative_to(raw).parts[0],
                         "source": rel, "vertices": int(len(vertices)), "points": args.points})
        except Exception as exc:
            failures.append({"source": rel, "error": str(exc)})
    if len(arrays) < 30:
        raise RuntimeError(f"Only {len(arrays)} valid detailed tiles; wait for more data")
    order = np.arange(len(rows)); rng = np.random.default_rng(2026); rng.shuffle(order)
    n = len(order); a, b = int(n * .60), int(n * .80)
    splits = np.empty(n, dtype="U5"); splits[order[:a]]="train"; splits[order[a:b]]="val"; splits[order[b:]]="test"
    for row, split in zip(rows, splits): row["split"] = str(split)
    np.savez_compressed(out / "points.npz", points=np.stack(arrays), ids=np.array([r["id"] for r in rows]))
    manifest = {"dataset":"Hong Kong 3D Visualisation Map textured tile-based models",
                "representation":"normalized 2048-point samples from B3DM mesh tiles",
                "models":rows, "failures":failures}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"valid":len(rows),"failures":len(failures),"splits":{s:int(sum(x["split"]==s for x in rows)) for s in ("train","val","test")},"out":str(out)}))


if __name__ == "__main__": main()
