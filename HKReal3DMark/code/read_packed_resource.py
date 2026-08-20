#!/usr/bin/env python3
"""Extract one retained JSON or B3DM resource from the packed data store."""
import argparse
import sqlite3
import urllib.parse
import zipfile
import zlib
from pathlib import Path
from packed_paths import inventory_path


def canonical(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="API resource URL without an API key")
    parser.add_argument("--root", default="/Volumes/SANDISK-ELE/HKReal3DMarkData/packed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    db = sqlite3.connect(inventory_path(root), timeout=60)
    row = db.execute(
        "select status,type,content,archive,member from urls where url=?", (canonical(args.url),)
    ).fetchone()
    db.close()
    if row is None:
        raise SystemExit("resource is not present in the inventory")
    status, kind, content, archive, member = row
    if status != "done":
        raise SystemExit(f"resource is not available (status={status})")
    if kind == "json":
        if content is None:
            raise SystemExit("JSON body has not been retained")
        data = zlib.decompress(content)
    else:
        if not archive or not member:
            raise SystemExit("payload archive mapping has not been recorded")
        with zipfile.ZipFile(root / archive) as packed:
            data = packed.read(member)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"wrote {len(data)} bytes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
