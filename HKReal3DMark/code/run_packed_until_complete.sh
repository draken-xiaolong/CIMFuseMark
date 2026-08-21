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
packed = Path(os.environ['HK3D_DATA_ROOT']) / 'packed'
pointer = packed / 'inventory.pointer'
db_path = Path(os.environ.get('HK3D_DB_PATH') or (pointer.read_text().strip() if pointer.exists() else packed / 'inventory.sqlite'))
db = sqlite3.connect(db_path)
selected = os.environ.get('HK3D_SELECTED_ONLY', '0').lower() in {'1', 'true', 'yes'}
if selected:
    total, terminal = db.execute("select count(*),sum(u.status in ('done','missing')) from payload_selection s join urls u on u.url=s.url").fetchone()
else:
    total, terminal = db.execute("select count(*),sum(status in ('done','missing')) from urls").fetchone()
raise SystemExit(0 if total == terminal else 1)
PY
  then
    python3 "$SCRIPT_DIR/audit_packed_download.py" --root "$(python3 - "$ENV_FILE" <<'PY'
import os, sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        key, value = line.split('=', 1); os.environ.setdefault(key.strip(), value.strip())
print(os.environ['HK3D_DATA_ROOT'])
PY
)" >>"$LOG_FILE" 2>&1
    exit 0
  fi
  sleep 30
done
