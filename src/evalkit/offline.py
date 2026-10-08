"""Offline evals: run a dataset through the current agent, judge every output."""
from __future__ import annotations

import glob
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .judge import load_judges
from .store import save_result, summarize
from .version import agent_version


def load_agent(spec: str):
    if ":" not in spec:
        raise ValueError("agent must look like 'module:function' in evalkit.yaml")
    mod, fn = spec.split(":", 1)
    sys.path.insert(0, ".")
    return getattr(importlib.import_module(mod), fn)


def load_dataset(path: str) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", f"ex{i}")
    return rows, hashlib.sha256(raw).hexdigest()[:10]


def run_offline(cfg: Config, version: str | None = None) -> list[tuple[dict, str]]:
    """Returns [(result, saved_path)] one per dataset."""
    agent = load_agent(cfg.agent)
    judges = load_judges(cfg.judges_dir, "offline")
    if not judges:
        raise ValueError(f"No offline judges found in {cfg.judges_dir}")
    version = version or agent_version(cfg.agent_name)
    out = []
    for pattern in cfg.datasets:
        for path in sorted(glob.glob(pattern)):
            rows, dhash = load_dataset(path)
            dname = Path(path).stem
            scores = []
            for ex in rows:
                for rep in range(cfg.repeats):
                    res = agent(ex["input"])
                    output, steps = (res["output"], res.get("steps")) if isinstance(res, dict) else (res, None)
                    for j in judges:
                        if j.requires_reference and ex.get("reference") is None:
                            continue
                        g = j.grade(input=ex["input"], output=output, reference=ex.get("reference"), steps=steps)
                        scores.append({"example_id": ex["id"], "repeat": rep, "judge": j.id, "output": output, **g})
            ts = datetime.now(timezone.utc)
            result = {
                "kind": "offline", "agent": cfg.agent_name, "timestamp": ts.isoformat(), "agent_version": version,
                "dataset": dname, "dataset_hash": dhash,
                "judges": {j.id: j.hash for j in judges},
                "scores": scores, "summary": summarize(scores),
            }
            name = f"{cfg.agent_name}__{version}__{dname}__{ts.strftime('%Y%m%dT%H%M%S%f')}".replace("/", "-")
            out.append((result, str(save_result(cfg.results_dir, "offline", name, result))))
    if not out:
        raise ValueError(f"No datasets matched {cfg.datasets}")
    return out
