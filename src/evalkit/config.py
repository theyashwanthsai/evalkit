from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    agent: str = ""
    traces_dir: str = ".traces"
    results_dir: str = ".evals/results"
    datasets: list[str] = field(default_factory=lambda: ["evals/datasets/*.jsonl"])
    judges_dir: str = "evals/judges"
    repeats: int = 1
    regression_threshold: float = 0.2
    sample_size: int = 20
    window_hours: float = 24
    include_errors: int = 3


def load_config(path: str | Path = "evalkit.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config()
    d = yaml.safe_load(p.read_text()) or {}
    off, on = d.get("offline", {}), d.get("online", {})
    c = Config()
    for k in ("agent", "traces_dir", "results_dir", "datasets", "judges_dir"):
        if k in d:
            setattr(c, k, d[k])
    c.repeats = off.get("repeats", c.repeats)
    c.regression_threshold = off.get("regression_threshold", c.regression_threshold)
    c.sample_size = on.get("sample_size", c.sample_size)
    c.window_hours = on.get("window_hours", c.window_hours)
    c.include_errors = on.get("include_errors", c.include_errors)
    return c
