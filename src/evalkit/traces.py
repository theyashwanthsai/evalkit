"""Traces live in the repo under `.traces/`, one JSON file per trace, tagged with the git version."""
from __future__ import annotations

import json
import random
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import settings
from .version import agent_version


@dataclass
class RunCtx:
    id: str
    run_id: str
    parent_id: str | None
    used: bool = False


_ctx: ContextVar[RunCtx | None] = ContextVar("evalkit_run", default=None)


@contextmanager
def run():
    """Wrap an agent's serve(). Agents called inside it (which also use run()) get the same run_id
    and this agent's trace id as their parent_id, so a multi-agent request can be stitched together."""
    cur, tid = _ctx.get(), uuid.uuid4().hex[:12]
    ctx = RunCtx(tid, cur.run_id if cur else tid, cur.id if cur else None)
    token = _ctx.set(ctx)
    try:
        yield ctx
    finally:
        _ctx.reset(token)


def record_trace(
    input,
    output,
    *,
    steps: list | None = None,
    error: str | None = None,
    metadata: dict | None = None,
    agent: str | None = None,
    version: str | None = None,
) -> dict:
    """Call this from your agent in production to write a trace."""
    agent = agent or settings.get_agent()
    ctx = _ctx.get()
    if ctx and not ctx.used:
        ctx.used, (tid, run_id, parent_id) = True, (ctx.id, ctx.run_id, ctx.parent_id)
    else:
        tid = uuid.uuid4().hex[:12]
        run_id, parent_id = (ctx.run_id, ctx.parent_id) if ctx else (tid, None)
    trace = {
        "id": tid,
        "run_id": run_id,
        "parent_id": parent_id,
        "agent": agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_version": version or agent_version(agent),
        "input": input,
        "output": output,
        "steps": steps or [],
        "error": error,
        "metadata": metadata or {},
    }
    redact = settings.get_redact()
    if redact:
        trace = redact(trace)
    settings.get_sink().write(trace)
    return trace


def load_traces(traces_dir: str | Path = ".traces", agent: str | None = None) -> list[dict]:
    """Read traces (.json and .jsonl). With `agent`, only that agent's folder/traces."""
    root = Path(traces_dir)
    if agent:
        root = root / agent
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
    return [t for t in traces if not agent or t.get("agent", "default") == agent]


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
