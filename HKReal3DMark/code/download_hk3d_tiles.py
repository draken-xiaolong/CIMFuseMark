#!/usr/bin/env python3
"""Resumable crawler/downloader for LandsD textured Cesium 3D Tiles."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT_URL = "https://data.map.gov.hk/api/3d-data/3dtiles/f2/tileset.json"
USER_AGENT = "HKReal3DMark-academic-download/1.0"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class Downloader:
    def __init__(self, key: str, root: Path, workers: int, retries: int):
        self.key = key
        self.root = root
        self.workers = workers
        self.retries = retries
        self.tasks: queue.Queue[str | None] = queue.Queue()
        self.seen: set[str] = set()
        self.lock = threading.Lock()
        self.started = time.time()
        self.files = self.bytes = self.skipped = self.failed = self.json_files = 0
        self.failures: list[dict[str, str]] = []
        self.state_path = root.parent.parent / "metadata" / "download_state.json"

    def authenticated(self, url: str) -> str:
        split = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(split.query)
        query["key"] = [self.key]
        return urllib.parse.urlunsplit(
            (split.scheme, split.netloc, split.path, urllib.parse.urlencode(query, doseq=True), "")
        )

    @staticmethod
    def identity(url: str) -> str:
        split = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit((split.scheme, split.netloc, split.path, "", ""))

    def local_path(self, url: str) -> Path:
        path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
        marker = "/api/3d-data/3dtiles/f2/"
        relative = path.split(marker, 1)[1] if marker in path else path.lstrip("/")
        target = (self.root / relative).resolve()
        if self.root.resolve() not in target.parents and target != self.root.resolve():
            raise ValueError(f"Unsafe output path: {target}")
        return target

    def enqueue(self, url: str) -> None:
        clean = self.identity(url)
        with self.lock:
            if clean in self.seen:
                return
            self.seen.add(clean)
        self.tasks.put(clean)

    def download(self, url: str, target: Path) -> bool:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            with self.lock:
                self.skipped += 1
            return True
        temporary = target.with_name(target.name + ".part")
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.authenticated(url), headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                size = temporary.stat().st_size
                temporary.replace(target)
                with self.lock:
                    self.files += 1
                    self.bytes += size
                return True
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                if attempt == self.retries:
                    with self.lock:
                        self.failed += 1
                        self.failures.append({"url": url, "error": str(error)})
                    return False
                time.sleep(min(30, 2 ** attempt))
        return False

    def discover(self, url: str, target: Path) -> None:
        if target.suffix.lower() != ".json":
            return
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            with self.lock:
                self.json_files += 1
            stack = [payload.get("root", payload)]
            while stack:
                node = stack.pop()
                stack.extend(node.get("children", []))
                content = node.get("content") or {}
                uri = content.get("uri") or content.get("url")
                if uri:
                    self.enqueue(urllib.parse.urljoin(url, uri))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
            with self.lock:
                self.failed += 1
                self.failures.append({"url": url, "error": f"parse: {error}"})

    def write_state(self) -> None:
        with self.lock:
            state = {
                "source": "Hong Kong Lands Department 3D Visualisation Map API",
                "dataset": "Textured tile-based models (f2)",
                "root_url": ROOT_URL,
                "output_root": str(self.root),
                "workers": self.workers,
                "discovered_urls": len(self.seen),
                "downloaded_files_this_run": self.files,
                "downloaded_bytes_this_run": self.bytes,
                "skipped_existing_this_run": self.skipped,
                "json_files_processed": self.json_files,
                "failures": self.failures[-100:],
                "failure_count": self.failed,
                "elapsed_seconds": time.time() - self.started,
                "complete": False,
            }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".json.part")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def worker(self) -> None:
        while True:
            url = self.tasks.get()
            try:
                if url is None:
                    return
                target = self.local_path(url)
                if self.download(url, target):
                    self.discover(url, target)
                with self.lock:
                    finished = self.files + self.skipped + self.failed
                if finished and finished % 250 == 0:
                    self.write_state()
                    print(json.dumps({
                        "processed": finished, "discovered": len(self.seen),
                        "downloaded_GiB": round(self.bytes / 1024 ** 3, 3),
                        "failed": self.failed,
                    }), flush=True)
            finally:
                self.tasks.task_done()

    def run(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.workers)]
        for thread in threads:
            thread.start()
        self.enqueue(ROOT_URL)
        self.tasks.join()
        for _ in threads:
            self.tasks.put(None)
        for thread in threads:
            thread.join()
        self.write_state()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state["complete"] = self.failed == 0
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps(state, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(Path(__file__).resolve().parents[1] / ".env"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    load_env(Path(args.env))
    key = os.environ.get("HK3D_API_KEY", "").strip()
    data_root = Path(os.environ.get("HK3D_DATA_ROOT", "")).expanduser()
    if not key:
        raise SystemExit("HK3D_API_KEY is not configured")
    if not str(data_root):
        raise SystemExit("HK3D_DATA_ROOT is not configured")
    output = data_root / "raw" / "tiles_f2"
    Downloader(key, output, max(1, args.workers), max(1, args.retries)).run()


if __name__ == "__main__":
    main()
