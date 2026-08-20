#!/bin/sh
set -eu

ENV_FILE=${1:-"$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)/.env"}
LOG_FILE=${2:-"/Volumes/SANDISK-ELE/HKReal3DMarkData/metadata/packed_download.log"}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORKERS=${HK3D_WORKERS:-32}

# Allow an older foreground downloader to finish its current ZIP shard cleanly.
while pgrep -f '[d]ownload_hk3d_packed.py' >/dev/null 2>&1; do sleep 60; done

# A run can end with transient DNS/SSL failures still queued. Restarting is safe:
# SQLite supplies only unfinished URLs and each run creates new ZIP64 shards.
while :; do
  python3 "$SCRIPT_DIR/download_hk3d_packed.py" --env "$ENV_FILE" --workers "$WORKERS" --shard-size 10000 >>"$LOG_FILE" 2>&1
  if python3 - "$ENV_FILE" <<'PY'
import os, sqlite3, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        key, value = line.split('=', 1); os.environ.setdefault(key.strip(), value.strip())
db = sqlite3.connect(Path(os.environ['HK3D_DATA_ROOT']) / 'packed' / 'inventory.sqlite')
total, terminal = db.execute("select count(*),sum(status in ('done','missing')) from urls").fetchone()
raise SystemExit(0 if total == terminal else 1)
PY
  then exit 0
  fi
  sleep 30
done
