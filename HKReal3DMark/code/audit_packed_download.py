#!/usr/bin/env python3
"""Strictly audit a completed packed Hong Kong 3D Tiles download."""
import argparse
import json
import sqlite3
import time
import zipfile
import zlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Volumes/SANDISK-ELE/HKReal3DMarkData")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    packed = root / "packed"
    report_path = Path(args.report) if args.report else root / "metadata" / "packed_audit.json"
    db = sqlite3.connect(packed / "inventory.sqlite", timeout=60)
    total, done, missing, queued, payload_done = db.execute(
        """select count(*), sum(status='done'), sum(status='missing'),
                  sum(status not in ('done','missing')),
                  sum(type='payload' and status='done') from urls"""
    ).fetchone()
    expected_payload_bytes = db.execute(
        "select coalesce(sum(size),0) from urls where type='payload' and status='done'"
    ).fetchone()[0]
    json_done, json_stored = db.execute(
        "select sum(type='json' and status='done'),sum(type='json' and status='done' and content is not null) from urls"
    ).fetchone()
    bad_json = []
    for url, content in db.execute("select url,content from urls where type='json' and status='done'"):
        try:
            json.loads(zlib.decompress(content))
        except Exception as exc:
            if len(bad_json) < 20:
                bad_json.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    db.close()

    shards = sorted(packed.glob("payload_*.zip"))
    shard_rows = []
    zip_entries = 0
    b3dm_entries = 0
    zip_uncompressed_bytes = 0
    valid = queued == 0
    for path in shards:
        row = {"name": path.name, "archive_bytes": path.stat().st_size}
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                row["entries"] = len(infos)
                row["uncompressed_bytes"] = sum(info.file_size for info in infos)
                row["crc_error"] = archive.testzip()
                row["b3dm_entries"] = sum(info.filename.lower().endswith(".b3dm") for info in infos)
                bad_magic = []
                for info in infos:
                    if info.filename.lower().endswith(".b3dm"):
                        with archive.open(info) as stream:
                            if stream.read(4) != b"b3dm":
                                bad_magic.append(info.filename)
                                if len(bad_magic) >= 20:
                                    break
                row["bad_b3dm_magic"] = bad_magic
                zip_entries += row["entries"]
                b3dm_entries += row["b3dm_entries"]
                zip_uncompressed_bytes += row["uncompressed_bytes"]
                valid &= row["crc_error"] is None and not bad_magic
        except (OSError, zipfile.BadZipFile) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            valid = False
        shard_rows.append(row)

    valid &= b3dm_entries == (payload_done or 0)
    valid &= zip_uncompressed_bytes == expected_payload_bytes
    valid &= (json_done or 0) == (json_stored or 0) and not bad_json
    report = {
        "schema_version": 1,
        "audited_unix_time": time.time(),
        "valid": bool(valid),
        "inventory": {
            "total": total or 0,
            "done": done or 0,
            "missing": missing or 0,
            "queued": queued or 0,
            "payload_done": payload_done or 0,
            "payload_bytes": expected_payload_bytes,
            "json_done": json_done or 0,
            "json_stored": json_stored or 0,
            "bad_json": bad_json,
        },
        "archives": {
            "shards": len(shards),
            "entries": zip_entries,
            "b3dm_entries": b3dm_entries,
            "uncompressed_bytes": zip_uncompressed_bytes,
            "details": shard_rows,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
