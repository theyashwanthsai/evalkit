"""Online evals: judge a sample of real production traces from .traces/."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from .config import Config
from .judge import load_judges
from .store import load_results, save_result, summarize
from .traces import load_traces, sample_traces


def run_online(cfg: Config, seed: int | None = None) -> tuple[dict, str] | None:
    judges = load_judges(cfg.judges_dir, "online")
    if not judges:
        raise ValueError(f"No online judges found in {cfg.judges_dir}")
    already = {s["trace_id"] for r in load_results(cfg.results_dir, "online") for s in r["scores"]}
    agent = cfg.agent_name
    traces = load_traces(cfg.traces_dir, agent)
    known = [t for t in traces if t.get("agent_version", "unknown") != "unknown"]
    if len(known) < len(traces):
        print(f"evalkit: skipping {len(traces) - len(known)} trace(s) with unknown agent_version "
              "(set EVALKIT_VERSION at deploy time)", file=sys.stderr)
    picked = sample_traces(
        known, cfg.sample_size,
        window_hours=cfg.window_hours, include_errors=cfg.include_errors,
        seed=seed, exclude_ids=already,
    )
    if not picked:
        return None
    scores = []
    for t in picked:
        for j in judges:
            g = j.grade(input=t.get("input"), output=t.get("output"), steps=t.get("steps"))
            scores.append({"trace_id": t["id"], "agent": agent, "agent_version": t.get("agent_version", "unknown"),
                           "judge": j.id, "errored": bool(t.get("error")), **g})
    ts = datetime.now(timezone.utc)
    by_version = {}
    for v in {s["agent_version"] for s in scores}:
        by_version[v] = summarize([s for s in scores if s["agent_version"] == v])
    result = {
        "kind": "online", "agent": agent, "timestamp": ts.isoformat(),
        "judges": {j.id: j.hash for j in judges},
        "sampled": len(picked), "scores": scores, "summary_by_version": by_version,
    }
    return result, str(save_result(cfg.results_dir, "online", f"{agent}__{ts.strftime('%Y%m%dT%H%M%S%f')}".replace("/", "-"), result))
