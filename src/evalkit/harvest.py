"""Turn low-scoring or errored online traces into offline dataset entries."""
from __future__ import annotations

import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .store import load_results
from .traces import load_traces


@dataclass
class HarvestCandidate:
    trace: dict
    judge_rows: list[dict]
    result_timestamp: str
    source_result: str

    @property
    def trace_id(self) -> str:
        return self.trace["id"]

    @property
    def errored(self) -> bool:
        return bool(self.trace.get("error")) or any(r.get("errored") for r in self.judge_rows)

    def judge_scores(self) -> dict[str, int | None]:
        return {r["judge"]: r.get("score") for r in self.judge_rows}

    def worst_score(self) -> int:
        nums = [s for s in self.judge_scores().values() if s is not None]
        return min(nums) if nums else 0


def load_harvested_trace_ids(datasets: list[str]) -> set[str]:
    """Trace ids already present in any dataset (by metadata or harvest id)."""
    seen: set[str] = set()
    for pattern in datasets:
        for path in glob.glob(pattern):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = (row.get("metadata") or {}).get("trace_id")
                if tid:
                    seen.add(tid)
                rid = row.get("id", "")
                if isinstance(rid, str) and rid.startswith("harvest-"):
                    seen.add(rid.removeprefix("harvest-"))
    return seen


def _collect_judgements(results_dir: str, agent: str) -> dict[str, list[dict]]:
    """Latest score row per (trace_id, judge) across all online result files."""
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for result in load_results(results_dir, "online"):
        if result.get("agent") != agent:
            continue
        rts = result.get("timestamp", "")
        for s in result["scores"]:
            key = (s["trace_id"], s["judge"])
            if key not in best or rts >= best[key][0]:
                best[key] = (rts, s)
    by_trace: dict[str, list[dict]] = {}
    for (tid, _j), (rts, row) in best.items():
        by_trace.setdefault(tid, []).append({**row, "_result_timestamp": rts})
    return by_trace


def _qualifies(rows: list[dict], trace: dict, min_score: int) -> bool:
    if trace.get("error"):
        return True
    if any(r.get("errored") for r in rows):
        return True
    for r in rows:
        sc = r.get("score")
        if sc is None or sc <= min_score:
            return True
    return False


def harvest_candidates(
    cfg: Config,
    *,
    min_score: int = 2,
    max_count: int = 20,
) -> list[HarvestCandidate]:
    agent = cfg.agent_name
    harvested = load_harvested_trace_ids(cfg.datasets)
    by_trace = _collect_judgements(cfg.results_dir, agent)
    if not by_trace:
        return []

    traces = {t["id"]: t for t in load_traces(cfg.traces_dir, agent)}
    out: list[HarvestCandidate] = []
    for tid, rows in by_trace.items():
        if tid in harvested:
            continue
        trace = traces.get(tid)
        if not trace:
            print(f"evalkit harvest: skipping {tid} (trace not found under {cfg.traces_dir})", file=sys.stderr)
            continue
        if not _qualifies(rows, trace, min_score):
            continue
        rts = max(r["_result_timestamp"] for r in rows)
        out.append(HarvestCandidate(trace=trace, judge_rows=rows, result_timestamp=rts, source_result=rts))

    out.sort(key=lambda c: (c.worst_score(), c.errored), reverse=False)
    return out[:max_count]


def _format_candidate(c: HarvestCandidate) -> str:
    lines = [
        "",
        f"--- trace {c.trace_id} ({c.trace.get('agent_version', '?')}) ---",
        f"input: {c.trace.get('input')!r}",
        f"output: {c.trace.get('output')!r}",
    ]
    if c.trace.get("error"):
        lines.append(f"error: {c.trace['error']!r}")
    for r in c.judge_rows:
        lines.append(
            f"  {r['judge']}: score={r.get('score')} passed={r.get('passed')} — {r.get('reasoning', '')[:120]}"
        )
    return "\n".join(lines)


def _prompt_reference(default: str) -> str | None:
    print(f"Reference answer [{default[:60]}{'…' if len(default) > 60 else ''}]:")
    ref = input("> ").strip()
    return ref or default


def review_interactive(candidates: list[HarvestCandidate]) -> list[dict]:
    entries: list[dict] = []
    for c in candidates:
        print(_format_candidate(c))
        while True:
            choice = input("Add to dataset? [y]es / [n]o / [q]uit: ").strip().lower()
            if choice in ("n", "no"):
                break
            if choice in ("q", "quit"):
                return entries
            if choice in ("y", "yes"):
                default_ref = str(c.trace.get("output", ""))
                reference = _prompt_reference(default_ref)
                inp = str(c.trace.get("input", ""))
                edit = input("Edit input? (press Enter to keep): ").strip()
                if edit:
                    inp = edit
                entries.append(_build_entry(c, inp, reference))
                break
            print("Please enter y, n, or q.")
    return entries


def _build_entry(c: HarvestCandidate, input_text: str, reference: str) -> dict:
    harvested_at = datetime.now(timezone.utc).isoformat()
    return {
        "id": f"harvest-{c.trace_id}",
        "input": input_text,
        "reference": reference,
        "metadata": {
            "harvested_from": c.source_result,
            "trace_id": c.trace_id,
            "agent_version": c.trace.get("agent_version"),
            "judge_scores": c.judge_scores(),
            "errored": c.errored,
            "harvested_at": harvested_at,
        },
    }


def append_entries(path: str | Path, entries: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_harvest(
    cfg: Config,
    *,
    min_score: int = 2,
    max_count: int = 20,
    output: str = "evals/datasets/harvested.jsonl",
    auto: bool = False,
) -> int:
    candidates = harvest_candidates(cfg, min_score=min_score, max_count=max_count)
    if not candidates:
        print("No harvest candidates (run evalkit online first, or lower --min-score).")
        return 0

    print(f"Found {len(candidates)} candidate(s) (min_score={min_score}).")
    if auto:
        entries = [
            _build_entry(c, str(c.trace.get("input", "")), str(c.trace.get("output", "")))
            for c in candidates
        ]
    else:
        entries = review_interactive(candidates)

    if not entries:
        print("Nothing added.")
        return 0

    append_entries(output, entries)
    for e in entries:
        print(f"harvested {e['id']} -> {output}")
    return len(entries)
