from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint_io import CheckpointError, atomic_write, canonical, durable_directory, promote_temp, read_json, sync_directory, writer_lock
from .checkpoint_state import Checkpoint, call_plan, digest, require, restore_judge
from .config import Config
from .judge import load_judges
from .store import summarize


ENV_CONFIGURATION = ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "ANTHROPIC_BASE_URL")


def prepare_manifest(cfg: Config, version: str | None, evaluation_version: str | None) -> dict:
    require(isinstance(version, str) and bool(version.strip()), "Checkpointing requires an explicit --version identifying the agent implementation/configuration")
    require(isinstance(evaluation_version, str) and bool(evaluation_version.strip()), "Checkpointing requires --evaluation-version identifying provider/custom evaluation configuration")
    require(type(cfg.repeats) is int and cfg.repeats > 0, "Checkpoint repeats must be a positive integer")
    judges = load_judges(cfg.judges_dir, "offline")
    require(bool(judges), f"No offline judges found in {cfg.judges_dir}")
    require(len({j.id for j in judges}) == len(judges), "Checkpoint judges need unique name/version identities")
    datasets = []
    for pattern in cfg.datasets:
        for path in sorted(glob.glob(pattern)):
            raw = Path(path).read_bytes()
            rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            require(all(isinstance(row, dict) and "input" in row for row in rows), f"Invalid dataset {path}: each example needs input")
            rows = [{**row, "id": row.get("id", f"ex{i}")} for i, row in enumerate(rows)]
            short_hash = hashlib.sha256(raw).hexdigest()[:10]
            datasets.append({
                "path": str(Path(path).resolve()), "name": Path(path).stem,
                "hash": hashlib.sha256(raw).hexdigest(), "short_hash": short_hash, "rows": rows,
            })
    require(bool(datasets), f"No datasets matched {cfg.datasets}")
    return json.loads(canonical({
        "agent": cfg.agent, "agent_name": cfg.agent_name, "agent_version": version,
        "evaluation_version": evaluation_version, "repeats": cfg.repeats,
        "regression_threshold": cfg.regression_threshold,
        "results_dir": str(Path(cfg.results_dir).resolve()),
        "checkpoints_dir": str(Path(cfg.checkpoints_dir).resolve()), "datasets": datasets,
        "judges": [{**{key: value for key, value in asdict(j).items() if key != "raw"}, "definition_hash": hashlib.sha256(j.raw.encode()).hexdigest()} for j in judges],
        "provider_configuration": {key: digest(os.environ.get(key)) for key in ENV_CONFIGURATION},
        "provider_defaults": {"temperature": 0, "anthropic_max_tokens": 1024, "implementation": 1},
    }))


def checkpoint_result(state: dict, dataset_index: int) -> dict:
    manifest = state["manifest"]
    dataset = manifest["datasets"][dataset_index]
    scores = []
    for key, kind, di, ei, repeat, ji in call_plan(manifest):
        if di != dataset_index or kind != "judge":
            continue
        example = dataset["rows"][ei]
        agent_key = key.rsplit("/", 1)[0] + "/agent"
        scores.append({
            "example_id": example["id"], "repeat": repeat,
            "judge": f"{manifest['judges'][ji]['name']}@v{manifest['judges'][ji]['version']}",
            "output": state["calls"][agent_key]["output"], **state["calls"][key]["grade"],
        })
    return {
        "kind": "offline", "agent": manifest["agent_name"], "timestamp": state["timestamp"],
        "agent_version": manifest["agent_version"], "dataset": dataset["name"],
        "dataset_hash": dataset["short_hash"],
        "judges": {f"{j['name']}@v{j['version']}": j["definition_hash"][:10] for j in manifest["judges"]},
        "scores": scores, "summary": summarize(scores),
    }


def publication_path(state: dict, dataset_index: int) -> Path:
    namespace = digest(state["manifest"]["checkpoints_dir"])[:16]
    return Path(state["manifest"]["results_dir"]) / "offline" / f"checkpoint-{namespace}-{state['run_id']}-d{dataset_index}.json"


def validate_publications(state: dict) -> None:
    plan = call_plan(state["manifest"])
    for di, published in enumerate(state["published"]):
        path = publication_path(state, di)
        temp = path.with_suffix(path.suffix + ".tmp")
        complete = all(state["calls"].get(slot[0], {}).get("status") == "done" for slot in plan if slot[2] == di)
        require(not published or path.exists(), f"Published result missing at {path}; restore it before resuming")
        if path.exists() or temp.exists():
            require(complete, f"Result exists for incomplete dataset at {path}; preserve and inspect files")
            expected = checkpoint_result(state, di)
            if path.exists():
                require(read_json(path) == expected, f"Conflicting result at {path}; not overwritten")
            if temp.exists():
                require(read_json(temp) == expected, f"Invalid or stale publication temp at {temp}; not promoted")


def publish(checkpoint: Checkpoint, dataset_index: int) -> tuple[dict, str]:
    state = checkpoint.state
    result = checkpoint_result(state, dataset_index)
    path = publication_path(state, dataset_index)
    durable_directory(path.parent)
    temp = path.with_suffix(path.suffix + ".tmp")
    if not path.exists():
        if temp.exists():
            require(read_json(temp) == result, f"Invalid publication temp at {temp}")
            promote_temp(temp, path)
        else:
            atomic_write(path, result)
    else:
        require(read_json(path) == result, f"Conflicting result at {path}")
        if temp.exists():
            require(read_json(temp) == result, f"Conflicting publication temp at {temp}")
            temp.unlink()
            sync_directory(path.parent)
    if not state["published"][dataset_index]:
        checkpoint.advance(published=[done or i == dataset_index for i, done in enumerate(state["published"])])
    return result, str(path)


def execute_call(checkpoint: Checkpoint, slot: tuple, agent, judges: list) -> None:
    key, kind, di, ei, repeat, ji = slot
    saved = checkpoint.state["calls"].get(key)
    if saved and saved["status"] == "done":
        return
    if saved is None:
        checkpoint.save_call(key, {"status": "pending"})
        saved = checkpoint.state["calls"][key]
    example = checkpoint.manifest["datasets"][di]["rows"][ei]
    if kind == "agent":
        response = agent(json.loads(canonical(example["input"])))
        output, steps = (response["output"], response.get("steps")) if isinstance(response, dict) else (response, None)
        call = json.loads(canonical({"status": "done", "output": output, "steps": steps}))
        checkpoint.save_call(key, call)
        return
    judge = judges[ji]
    if saved["status"] == "pending":
        agent_call = checkpoint.state["calls"][key.rsplit("/", 1)[0] + "/agent"]
        raw = judge.request(input=example["input"], output=agent_call["output"], reference=example.get("reference"), steps=agent_call["steps"])
        require(isinstance(raw, str), "Judge provider must return raw text")
        checkpoint.save_call(key, {"status": "raw", "raw": raw})
        saved = checkpoint.state["calls"][key]
    checkpoint.save_call(key, {**saved, "status": "done", "grade": judge.parse_response(saved["raw"])})


def run_checkpointed(cfg: Config, version: str | None, *, run_id: str | None, resume: str | None, repeat_uncertain: bool, evaluation_version: str | None) -> list[tuple[dict, str]]:
    require(bool(run_id) != bool(resume), "Choose exactly one of --checkpoint RUN_ID or --resume RUN_ID")
    require(not repeat_uncertain or bool(resume), "--repeat-uncertain requires --resume")
    identity = resume or run_id
    require(isinstance(identity, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identity) is not None, "Run identity must contain 1-80 letters, digits, underscores or hyphens")
    root = Path(cfg.checkpoints_dir).resolve()
    results = Path(cfg.results_dir).resolve()
    require(root != results and results not in root.parents and root not in results.parents, "Checkpoint and completed-result directories must be separate, non-overlapping directories")
    manifest = prepare_manifest(cfg, version, evaluation_version or cfg.evaluation_version)
    with writer_lock(root, identity):
        directory = root / identity
        if resume:
            checkpoint = Checkpoint(directory / "state.json", identity, manifest, validate_artifacts=validate_publications)
        else:
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise CheckpointError(f"Run '{identity}' already exists; use --resume or choose a new run identity") from exc
            sync_directory(root)
            initial = {
                "schema_version": 1, "run_id": identity, "revision": 0, "previous": None,
                "manifest": manifest, "calls": {}, "published": [False] * len(manifest["datasets"]),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            checkpoint = Checkpoint(directory / "state.json", identity, manifest, initial, validate_artifacts=validate_publications)
        pending = [key for key, call in checkpoint.state["calls"].items() if call["status"] == "pending"]
        if pending:
            require(repeat_uncertain, f"Uncertain pending call {pending[0]} in run '{identity}'. Inspect side effects/cost, then explicitly use --resume {identity} --repeat-uncertain to repeat unresolved calls")
            warnings.warn(f"Repeating {len(pending)} uncertain call(s) in run '{identity}'; possible repeated side effects and provider cost", RuntimeWarning, stacklevel=2)
        from .offline import load_agent
        needs_agent = any(slot[1] == "agent" and checkpoint.state["calls"].get(slot[0], {}).get("status") != "done" for slot in call_plan(manifest))
        agent = load_agent(cfg.agent) if needs_agent else None
        judges = [restore_judge(definition) for definition in manifest["judges"]]
        plan = call_plan(manifest)
        out = []
        for di in range(len(manifest["datasets"])):
            for slot in plan:
                if slot[2] == di:
                    execute_call(checkpoint, slot, agent, judges)
            out.append(publish(checkpoint, di))
        return out
