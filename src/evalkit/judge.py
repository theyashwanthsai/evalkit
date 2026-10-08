"""A judge = prompt + output schema + model + version. Same definition powers offline and online."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import providers

JSON_SUFFIX = (
    '\n\nRespond with ONLY a JSON object: {"score": <integer between %d and %d>, '
    '"reasoning": "<one or two sentences>"}'
)


@dataclass
class Judge:
    name: str
    version: int
    prompt: str
    model: str = "claude-haiku-4-5-20251001"
    provider: str = "anthropic"
    modes: list[str] = field(default_factory=lambda: ["offline", "online"])
    requires_reference: bool = False
    scale: tuple[int, int] = (1, 5)
    pass_threshold: int = 4
    raw: str = ""

    @property
    def hash(self) -> str:
        """Content hash: catches prompt edits made without bumping `version`."""
        return hashlib.sha256(self.raw.encode()).hexdigest()[:10]

    @property
    def id(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, *, input, output, reference=None, steps=None) -> str:
        def s(x):
            return x if isinstance(x, str) else json.dumps(x, default=str, indent=2)
        body = (
            self.prompt.replace("{input}", s(input))
            .replace("{output}", s(output))
            .replace("{reference}", s(reference) if reference is not None else "")
            .replace("{steps}", s(steps or []))
        )
        return body + JSON_SUFFIX % self.scale

    def grade(self, *, input, output, reference=None, steps=None) -> dict:
        lo, hi = self.scale
        text = providers.complete(
            self.provider, self.model,
            self.render(input=input, output=output, reference=reference, steps=steps),
        )
        score, reasoning = parse_judgement(text, lo, hi)
        return {"score": score, "reasoning": reasoning, "passed": score is not None and score >= self.pass_threshold}


def parse_judgement(text: str, lo: int, hi: int) -> tuple[int | None, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            score = int(d["score"])
            if lo <= score <= hi:
                return score, str(d.get("reasoning", ""))
        except (ValueError, KeyError, TypeError):
            pass
    return None, f"UNPARSEABLE: {text[:200]}"


def load_judge(path: str | Path) -> Judge:
    raw = Path(path).read_text()
    d = yaml.safe_load(raw)
    scale = d.get("scale", [1, 5])
    return Judge(
        name=d["name"], version=int(d["version"]), prompt=d["prompt"],
        model=d.get("model", "claude-haiku-4-5-20251001"),
        provider=d.get("provider", "anthropic"),
        modes=d.get("modes", ["offline", "online"]),
        requires_reference=bool(d.get("requires_reference", False)),
        scale=(scale[0], scale[1]),
        pass_threshold=int(d.get("pass_threshold", scale[1] - 1)),
        raw=raw,
    )


def load_judges(judges_dir: str | Path, mode: str) -> list[Judge]:
    judges = [load_judge(p) for p in sorted(Path(judges_dir).glob("*.y*ml"))]
    out = []
    for j in judges:
        if mode not in j.modes:
            continue
        if mode == "online" and j.requires_reference:
            raise ValueError(
                f"Judge '{j.name}' requires a reference answer but is enabled for online mode, "
                "where production traces have none. Remove 'online' from its modes."
            )
        out.append(j)
    return out
