#!/usr/bin/env python3
"""Create a reproducible, region-stratified payload download subset."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit


def region(url: str) -> str:
    rest = urlsplit(url).path.split("/f2/", 1)[-1]
    return rest.split("/", 1)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--additional", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    db = sqlite3.connect(a.db)
    counts = Counter()
    for (url,) in db.execute("select url from urls where type='payload' and status='queued'"):
        counts[region(url)] += 1
    total = sum(counts.values())
    # Proportional allocation with a small regional floor so small territories
    # are not erased by Hong Kong's highly imbalanced tile hierarchy.
    floor = min(500, a.additional // max(1, 2 * len(counts)))
    quotas = {r: min(n, floor) for r, n in counts.items()}
    remaining = a.additional - sum(quotas.values())
    weights = {r: max(0, n - quotas[r]) for r, n in counts.items()}
    weight_total = sum(weights.values())
    raw = {r: remaining * weights[r] / weight_total for r in counts}
    for r in counts:
        quotas[r] += int(raw[r])
    for r, _ in sorted(raw.items(), key=lambda x: (-(x[1] - int(x[1])), x[0]))[: a.additional - sum(quotas.values())]:
        quotas[r] += 1

    rngs = {r: random.Random(f"{a.seed}:{r}") for r in counts}
    seen = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    for (url,) in db.execute("select url from urls where type='payload' and status='queued' order by rowid"):
        r = region(url); seen[r] += 1; q = quotas[r]; bucket = samples[r]
        if len(bucket) < q:
            bucket.append(url)
        else:
            j = rngs[r].randrange(seen[r])
            if j < q:
                bucket[j] = url

    db.execute("drop table if exists payload_selection")
    db.execute("create table payload_selection(url text primary key, region text not null, seed integer not null)")
    db.executemany("insert into payload_selection values(?,?,?)", ((u, r, a.seed) for r in sorted(samples) for u in samples[r]))
    db.execute("create index if not exists payload_selection_region on payload_selection(region)")
    db.commit()
    selected = {r: len(samples[r]) for r in sorted(samples)}
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps({"seed": a.seed, "additional_target": a.additional, "queued_population": total,
                                     "strategy": "region-stratified deterministic reservoir sampling",
                                     "population_by_region": dict(sorted(counts.items())),
                                     "selected_by_region": selected}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": sum(selected.values()), "regions": len(selected), "summary": str(a.summary)}))


if __name__ == "__main__":
    main()
