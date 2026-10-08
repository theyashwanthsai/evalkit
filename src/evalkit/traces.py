"""Traces live in the repo under `.traces/`, one JSON file per trace, tagged with the git version."""
from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .version import agent_version


def record_trace(
    input,
    output,
    *,
    steps: list | None = None,
    error: str | None = None,
    metadata: dict | None = None,
    traces_dir: str | Path = ".traces",
    version: str | None = None,
) -> dict:
    """Call this from your agent in production to write a trace."""
    now = datetime.now(timezone.utc)
    trace = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": now.isoformat(),
        "agent_version": version or agent_version(),
        "input": input,
        "output": output,
        "steps": steps or [],
        "error": error,
        "metadata": metadata or {},
    }
    day_dir = Path(traces_dir) / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"{trace['id']}.json").write_text(json.dumps(trace, indent=2, default=str))
    return trace


def load_traces(traces_dir: str | Path = ".traces") -> list[dict]:
    root = Path(traces_dir)
    traces: list[dict] = []
    if not root.exists():
        return traces
    for p in sorted(root.rglob("*.json")):
        try:
            traces.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    for p in sorted(root.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return traces


def _ts(trace: dict) -> float:
    try:
        return datetime.fromisoformat(trace["timestamp"]).timestamp()
    except (KeyError, ValueError):
        return 0.0


def sample_traces(
    traces: list[dict],
    n: int,
    *,
    window_hours: float | None = 24,
    include_errors: int = 0,
    seed: int | None = None,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Random sample from the window, plus up to `include_errors` traces that errored
    (so rare failures aren't missed by uniform sampling)."""
    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()
    pool = [t for t in traces if t.get("id") not in exclude_ids]
    if window_hours is not None:
        cutoff = time.time() - window_hours * 3600
        pool = [t for t in pool if _ts(t) >= cutoff]
    errored = [t for t in pool if t.get("error")]
    rng.shuffle(errored)
    picked = errored[:include_errors]
    picked_ids = {t["id"] for t in picked}
    rest = [t for t in pool if t["id"] not in picked_ids]
    rng.shuffle(rest)
    return picked + rest[: max(0, n - len(picked))]
