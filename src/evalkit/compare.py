"""Compare offline runs: paired per-example deltas with a noise margin."""
from __future__ import annotations

from math import sqrt
from statistics import mean, stdev


def _per_example(result: dict, judge: str) -> dict[str, float]:
    by: dict[str, list[int]] = {}
    for s in result["scores"]:
        if s["judge"] == judge and s["score"] is not None:
            by.setdefault(s["example_id"], []).append(s["score"])
    return {k: mean(v) for k, v in by.items()}


def find_baseline(results: list[dict], current: dict, baseline_version: str | None = None) -> dict | None:
    """Latest result of another version on the same dataset (preferring identical dataset hash)."""
    cands = [r for r in results
             if r.get("agent", "default") == current.get("agent", "default")
             and r["dataset"] == current["dataset"] and r["agent_version"] != current["agent_version"]
             and r["timestamp"] < current["timestamp"]]
    if baseline_version:
        cands = [r for r in cands if r["agent_version"] == baseline_version]
    return cands[-1] if cands else None


def compare(current: dict, baseline: dict, threshold: float = 0.2, z: float = 1.96) -> dict:
    notes, rows, regressed = [], [], False
    if current["dataset_hash"] != baseline["dataset_hash"]:
        notes.append("Dataset changed since baseline; only examples present in both are compared.")
    for jid, h in current["judges"].items():
        if jid not in baseline["judges"]:
            notes.append(f"{jid}: not in baseline (judge version/name changed), skipped.")
            continue
        if baseline["judges"][jid] != h:
            notes.append(f"{jid}: judge content changed without a version bump, skipped.")
            continue
        a, b = _per_example(current, jid), _per_example(baseline, jid)
        deltas = [a[k] - b[k] for k in a if k in b]
        if not deltas:
            continue
        d = mean(deltas)
        se = stdev(deltas) / sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        reg = d < -threshold and (d + z * se) < 0   # drop is big AND statistically clear
        imp = d > threshold and (d - z * se) > 0
        regressed |= reg
        rows.append({"judge": jid, "baseline": round(mean(b[k] for k in a if k in b), 3),
                     "current": round(mean(a[k] for k in a if k in b), 3),
                     "delta": round(d, 3), "se": round(se, 3), "n": len(deltas),
                     "verdict": "REGRESSION" if reg else "improved" if imp else "no significant change"})
    return {"rows": rows, "regressed": regressed, "notes": notes}


def to_markdown(current: dict, baseline: dict, cmp: dict) -> str:
    out = [f"### evalkit: `{current['agent_version']}` vs `{baseline['agent_version']}` ({current['dataset']})", "",
           "| judge | baseline | current | Δ | ±SE | n | verdict |", "|---|---|---|---|---|---|---|"]
    for r in cmp["rows"]:
        out.append(f"| {r['judge']} | {r['baseline']} | {r['current']} | {r['delta']:+} | {r['se']} | {r['n']} | {r['verdict']} |")
    out += [""] + [f"> {n}" for n in cmp["notes"]]
    return "\n".join(out)
