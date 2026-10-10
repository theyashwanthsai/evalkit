"""Compare LLM judge scores to human labels in evals/human_labels.jsonl."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .config import Config
from .judge import load_judges
from .store import load_results


@dataclass
class LabelRow:
    judge: str
    human_score: int
    example_id: str | None = None
    trace_id: str | None = None
    note: str | None = None


@dataclass
class Comparison:
    label: LabelRow
    judge_score: int | None
    judge_passed: bool | None
    source: str  # offline | online | missing


def load_human_labels(path: str | Path) -> list[LabelRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Human labels not found: {p}")
    out: list[LabelRow] = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i}: invalid JSON: {e}") from e
        ex, tr = d.get("example_id"), d.get("trace_id")
        if bool(ex) == bool(tr):
            raise ValueError(f"{p}:{i}: set exactly one of example_id or trace_id")
        if "judge" not in d or "human_score" not in d:
            raise ValueError(f"{p}:{i}: need judge and human_score")
        out.append(
            LabelRow(
                judge=str(d["judge"]),
                human_score=int(d["human_score"]),
                example_id=str(ex) if ex else None,
                trace_id=str(tr) if tr else None,
                note=d.get("note"),
            )
        )
    return out


def _index_offline_scores(results_dir: str, agent: str) -> dict[tuple[str, str], dict]:
    """Latest judge row per (judge_id, example_id)."""
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for result in load_results(results_dir, "offline"):
        if result.get("agent") != agent:
            continue
        ts = result.get("timestamp", "")
        for s in result["scores"]:
            eid = s.get("example_id")
            if not eid:
                continue
            key = (s["judge"], eid)
            if key not in best or ts >= best[key][0]:
                best[key] = (ts, s)
    return {k: v[1] for k, v in best.items()}


def _index_online_scores(results_dir: str, agent: str) -> dict[tuple[str, str], dict]:
    """Latest judge row per (judge_id, trace_id)."""
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for result in load_results(results_dir, "online"):
        if result.get("agent") != agent:
            continue
        ts = result.get("timestamp", "")
        for s in result["scores"]:
            tid = s.get("trace_id")
            if not tid:
                continue
            key = (s["judge"], tid)
            if key not in best or ts >= best[key][0]:
                best[key] = (ts, s)
    return {k: v[1] for k, v in best.items()}


def compare_labels(cfg: Config, labels: list[LabelRow]) -> list[Comparison]:
    offline = _index_offline_scores(cfg.results_dir, cfg.agent_name)
    online = _index_online_scores(cfg.results_dir, cfg.agent_name)
    out: list[Comparison] = []
    for lab in labels:
        if lab.example_id:
            row = offline.get((lab.judge, lab.example_id))
            source = "offline" if row else "missing"
        else:
            row = online.get((lab.judge, lab.trace_id or ""))
            source = "online" if row else "missing"
        if row:
            out.append(
                Comparison(
                    label=lab,
                    judge_score=row.get("score"),
                    judge_passed=row.get("passed"),
                    source=source,
                )
            )
        else:
            out.append(Comparison(label=lab, judge_score=None, judge_passed=None, source="missing"))
    return out


def _pass_thresholds(judges_dir: str) -> dict[str, int]:
    out = {}
    for j in load_judges(judges_dir, "offline"):
        out[j.id] = j.pass_threshold
    for j in load_judges(judges_dir, "online"):
        out.setdefault(j.id, j.pass_threshold)
    return out


@dataclass
class JudgeStats:
    judge: str
    labels: int
    matched: int
    exact_match: float | None
    within_one: float | None
    pass_agreement: float | None
    mean_delta: float | None  # judge - human; positive => lenient


def _stats_for_judge(judge: str, rows: list[Comparison], thresholds: dict[str, int]) -> JudgeStats:
    subset = [r for r in rows if r.label.judge == judge]
    matched_rows = [r for r in subset if r.judge_score is not None]
    pt = thresholds.get(judge, 4)
    exact, within, pass_ok, deltas = [], [], [], []
    for r in matched_rows:
        hs, js = r.label.human_score, r.judge_score
        assert js is not None
        exact.append(hs == js)
        within.append(abs(hs - js) <= 1)
        deltas.append(js - hs)
        hp = hs >= pt
        if r.judge_passed is not None:
            pass_ok.append(hp == r.judge_passed)
    n = len(subset)
    m = len(matched_rows)
    return JudgeStats(
        judge=judge,
        labels=n,
        matched=m,
        exact_match=round(sum(exact) / m, 3) if m else None,
        within_one=round(sum(within) / m, 3) if m else None,
        pass_agreement=round(sum(pass_ok) / len(pass_ok), 3) if pass_ok else None,
        mean_delta=round(mean(deltas), 3) if deltas else None,
    )


def format_report(rows: list[Comparison], stats: list[JudgeStats]) -> str:
    lines = [
        "# evalkit calibrate",
        "",
        "Compare human labels to the latest stored judge scores (offline example_id, online trace_id).",
        "",
        "## Summary",
        "",
        "| judge | labels | matched | exact match | within 1 | pass agree | mean(judge−human) | bias |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in stats:
        bias = "—"
        if s.mean_delta is not None:
            if s.mean_delta > 0.25:
                bias = "lenient"
            elif s.mean_delta < -0.25:
                bias = "harsh"
            else:
                bias = "≈ aligned"
        lines.append(
            f"| {s.judge} | {s.labels} | {s.matched} | {s.exact_match} | {s.within_one} | "
            f"{s.pass_agreement} | {s.mean_delta} | {bias} |"
        )
    missing = [r for r in rows if r.source == "missing"]
    if missing:
        lines += ["", "## Missing judge scores (run offline/online first)", ""]
        for r in missing:
            ref = r.label.example_id or r.label.trace_id
            lines.append(f"- `{r.label.judge}` on `{ref}`")
    unparsed = [r for r in rows if r.judge_score is None and r.source != "missing"]
    if unparsed:
        lines += ["", "## Unparseable judge scores", ""]
        for r in unparsed:
            ref = r.label.example_id or r.label.trace_id
            lines.append(f"- `{r.label.judge}` on `{ref}`")
    lines += ["", "## Per label", "", "| ref | judge | human | judge | Δ | source |", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        ref = r.label.example_id or r.label.trace_id
        js = r.judge_score if r.judge_score is not None else "—"
        delta = (r.judge_score - r.label.human_score) if r.judge_score is not None else "—"
        lines.append(f"| {ref} | {r.label.judge} | {r.label.human_score} | {js} | {delta} | {r.source} |")
    return "\n".join(lines)


def run_calibrate(cfg: Config, *, labels_path: str = "evals/human_labels.jsonl") -> tuple[str, list[JudgeStats]]:
    labels = load_human_labels(labels_path)
    if not labels:
        raise ValueError(f"No labels in {labels_path}")
    rows = compare_labels(cfg, labels)
    judges = sorted({lab.judge for lab in labels})
    thresholds = _pass_thresholds(cfg.judges_dir)
    stats = [_stats_for_judge(j, rows, thresholds) for j in judges]
    return format_report(rows, stats), stats


def run_calibrate_cli(
    cfg: Config,
    *,
    labels_path: str = "evals/human_labels.jsonl",
    min_agreement: float | None = None,
    markdown: str | None = None,
) -> int:
    try:
        text, stats = run_calibrate(cfg, labels_path=labels_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    print(text)
    if markdown:
        Path(markdown).write_text(text)
    if min_agreement is not None:
        rates = [s.exact_match for s in stats if s.matched and s.exact_match is not None]
        if not rates:
            print("\nNo matched labels to check agreement.", file=sys.stderr)
            return 1
        worst = min(rates)
        if worst < min_agreement:
            print(
                f"\nLowest exact-match rate {worst} is below --min-agreement {min_agreement}.",
                file=sys.stderr,
            )
            return 1
    return 0
