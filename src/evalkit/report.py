"""Score history per agent version, offline and online."""
from __future__ import annotations

from .store import load_results, summarize


def build_report(results_dir: str) -> str:
    out = ["# evalkit report", "", "## Offline (per agent version, latest run)", "",
           "| version | dataset | judge | mean | pass rate | n |", "|---|---|---|---|---|---|"]
    latest = {}
    for r in load_results(results_dir, "offline"):
        latest[(r["agent_version"], r["dataset"])] = r
    for (v, d), r in sorted(latest.items(), key=lambda kv: kv[1]["timestamp"]):
        for j, s in r["summary"].items():
            out.append(f"| {v} | {d} | {j} | {s['mean']} | {s['pass_rate']} | {s['n']} |")
    out += ["", "## Online (production traces, pooled by agent version)", "",
            "| version | judge | mean | pass rate | n |", "|---|---|---|---|---|"]
    pooled: dict[str, list[dict]] = {}
    for r in load_results(results_dir, "online"):
        for s in r["scores"]:
            pooled.setdefault(s["agent_version"], []).append(s)
    for v, rows in pooled.items():
        for j, s in summarize(rows).items():
            out.append(f"| {v} | {j} | {s['mean']} | {s['pass_rate']} | {s['n']} |")
    return "\n".join(out)
