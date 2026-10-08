"""Where traces go. `LocalSink` writes files; add others by subclassing `Sink`."""
from __future__ import annotations

import atexit
import base64
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


class Sink:
    def write(self, trace: dict) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass


class LocalSink(Sink):
    """<dir>/<agent>/<YYYY-MM-DD>/<id>.json, one file per trace."""

    def __init__(self, directory: str | Path = ".traces"):
        self.dir = Path(directory)

    def write(self, trace: dict) -> None:
        day = trace["timestamp"][:10]
        d = self.dir / trace["agent"] / day
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{trace['id']}.json").write_text(json.dumps(trace, indent=2, default=str))


class GitHubSink(Sink):
    """Batches traces and commits them to a (separate, usually private) GitHub repo through the
    contents API. No checkout on the server. One .jsonl file per agent/day per flush:

        <agent>/<YYYY-MM-DD>/<host>-<timestamp>-<rand>.jsonl

    A background thread flushes every `flush_interval` seconds or once `batch_size` traces are
    queued, and again at exit. Failures never raise into the request path: traces stay queued and
    are retried on the next flush (use flush() yourself in serverless handlers).
    The token needs contents:write on `repo` only. `branch` must already exist (default: repo default).
    """

    MAX_QUEUE = 10_000

    def __init__(self, repo: str, branch: str | None = None, token: str | None = None,
                 batch_size: int = 20, flush_interval: float = 60.0,
                 api: str = "https://api.github.com"):
        self.repo, self.branch, self.api = repo, branch, api.rstrip("/")
        self.token = token or os.environ.get("EVALKIT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHubSink needs a token: set GITHUB_TOKEN (contents:write on the traces repo)")
        self.batch_size, self.flush_interval = batch_size, flush_interval
        self._buf: list[dict] = []
        self._lock, self._flush_lock = threading.Lock(), threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._host = socket.gethostname()
        atexit.register(self.flush)

    def write(self, trace: dict) -> None:
        with self._lock:
            self._buf.append(trace)
            if len(self._buf) > self.MAX_QUEUE:
                del self._buf[: len(self._buf) - self.MAX_QUEUE]
            full = len(self._buf) >= self.batch_size
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        if full:
            self._wake.set()

    def _loop(self) -> None:
        while True:
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            self.flush()

    def flush(self) -> None:
        with self._flush_lock:
            with self._lock:
                batch, self._buf = self._buf, []
            groups: dict[tuple, list[dict]] = {}
            for t in batch:
                groups.setdefault((t["agent"], t["timestamp"][:10]), []).append(t)
            stamp = time.strftime("%H%M%S", time.gmtime())
            failed: list[dict] = []
            for (agent, day), traces in groups.items():
                path = f"{agent}/{day}/{self._host}-{stamp}-{uuid.uuid4().hex[:6]}.jsonl"
                body = "\n".join(json.dumps(t, default=str) for t in traces) + "\n"
                try:
                    self._put(path, body, f"evalkit: {len(traces)} trace(s) for {agent}")
                except Exception as e:  # never break the request path
                    print(f"evalkit: GitHubSink flush failed ({e}); will retry", file=sys.stderr)
                    failed += traces
            if failed:
                with self._lock:
                    self._buf = failed + self._buf

    def _put(self, path: str, content: str, message: str) -> None:
        payload = {"message": message, "content": base64.b64encode(content.encode()).decode()}
        if self.branch:
            payload["branch"] = self.branch
        req = urllib.request.Request(
            f"{self.api}/repos/{self.repo}/contents/{urllib.parse.quote(path)}",
            data=json.dumps(payload).encode(), method="PUT",
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "evalkit"},
        )
        try:
            urllib.request.urlopen(req, timeout=15).close()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}") from e
