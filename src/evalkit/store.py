"""Results are plain JSON files in the repo: .evals/results/{offline,online}/*.json"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def save_result(results_dir: str, kind: str, name: str, result: dict) -> Path:
    d = Path(results_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(result, indent=2))
    return p


def load_results(results_dir: str, kind: str) -> list[dict]:
    d = Path(results_dir) / kind
    if not d.exists():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return sorted(out, key=lambda r: r.get("timestamp", ""))


def load_latest_online_result(results_dir: str, agent: str | None = None) -> dict | None:
    rows = load_results(results_dir, "online")
    if agent:
        rows = [r for r in rows if r.get("agent") == agent]
    return rows[-1] if rows else None


def summarize(scores: list[dict]) -> dict:
    """{judge_id: {mean, pass_rate, n}} ignoring unparseable (score=None)."""
    by: dict[str, list[dict]] = {}
    for s in scores:
        by.setdefault(s["judge"], []).append(s)
    out = {}
    for j, rows in by.items():
        valid = [r for r in rows if r["score"] is not None]
        out[j] = {
            "mean": round(mean(r["score"] for r in valid), 3) if valid else None,
            "pass_rate": round(sum(r["passed"] for r in valid) / len(valid), 3) if valid else None,
            "n": len(valid),
            "failed_to_parse": len(rows) - len(valid),
        }
    return out
