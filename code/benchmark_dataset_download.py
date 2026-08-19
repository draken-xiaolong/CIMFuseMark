#!/usr/bin/env python3
"""Measure an uncached PLATEAU source download with per-file audit records."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True, help="Must be an empty dedicated benchmark directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args(); destination = Path(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty benchmark directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")); rows = []
    aria2 = shutil.which("aria2c")
    if aria2 is None:
        raise RuntimeError("aria2c is required so the benchmark matches the dataset preparation path")
    total_started = time.perf_counter()
    for source in config["sources"]:
        suffix = ".zip" if source.get("kind") == "archive" else Path(source["url"]).suffix or ".gml"
        target = destination / f"{source['id']}{suffix}"
        started = time.perf_counter()
        subprocess.run([aria2, "--console-log-level=warn", "--allow-overwrite=true", "--auto-file-renaming=false",
                        "--file-allocation=none", f"--max-connection-per-server={args.connections}",
                        f"--split={args.connections}", f"--dir={destination}", f"--out={target.name}", source["url"]],
                       check=True)
        elapsed = time.perf_counter() - started
        rows.append({"id": source["id"], "region": source["region"], "url": source["url"],
                     "bytes": target.stat().st_size, "elapsed_seconds": elapsed,
                     "mib_per_second": target.stat().st_size / 2**20 / elapsed})
        print(json.dumps(rows[-1]), flush=True)
    report = {"protocol": "uncached sequential sources; eight connections per source via aria2c",
              "files": len(rows), "bytes": sum(row["bytes"] for row in rows),
              "elapsed_seconds": time.perf_counter() - total_started, "sources": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("files", "bytes", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
