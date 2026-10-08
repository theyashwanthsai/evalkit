"""Runtime settings for tracing: agent name, sink, redaction. Env vars and evalkit.yaml, or configure()."""
from __future__ import annotations

import os
from typing import Callable

from .config import load_config
from .sinks import GitHubSink, LocalSink, Sink

_state: dict = {"agent": None, "sink": None, "redact": None}
_cached: tuple[tuple, Sink] | None = None


def configure(*, agent: str | None = None, sink: Sink | None = None,
              redact: Callable[[dict], dict] | None = None) -> None:
    """Override settings in code. redact(trace) -> trace runs before anything is written."""
    for k, v in (("agent", agent), ("sink", sink), ("redact", redact)):
        if v is not None:
            _state[k] = v


def get_agent() -> str:
    return _state["agent"] or os.environ.get("EVALKIT_AGENT") or load_config().agent_name


def get_redact() -> Callable[[dict], dict] | None:
    return _state["redact"]


def get_sink() -> Sink:
    """Sink from configure(), else evalkit.yaml `traces:` / EVALKIT_TRACES_REPO, else local files."""
    global _cached
    if _state["sink"]:
        return _state["sink"]
    cfg = load_config()
    t = cfg.traces
    repo = os.environ.get("EVALKIT_TRACES_REPO") or (t.get("repo") if t.get("sink") == "github" else None)
    spec = (cfg.traces_dir, repo, t.get("branch"), t.get("batch_size"), t.get("flush_interval"))
    if _cached is None or _cached[0] != spec:
        if repo:
            kw = {k: t[k] for k in ("batch_size", "flush_interval") if k in t}
            sink: Sink = GitHubSink(repo, branch=t.get("branch"), **kw)
        else:
            sink = LocalSink(cfg.traces_dir)
        _cached = (spec, sink)
    return _cached[1]
