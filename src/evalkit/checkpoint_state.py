from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .checkpoint_io import CheckpointError, atomic_write, canonical, promote_temp, read_json
from .judge import Judge

SCHEMA_VERSION = 1


def restore_judge(definition: dict) -> Judge:
    fields = {key: value for key, value in definition.items() if key != "definition_hash"}
    return Judge(**{**fields, "scale": tuple(fields["scale"])})


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def call_plan(manifest: dict) -> list[tuple[str, str, int, int, int, int | None]]:
    slots = []
    for di, dataset in enumerate(manifest["datasets"]):
        for ei, example in enumerate(dataset["rows"]):
            for repeat in range(manifest["repeats"]):
                key = f"d{di}/e{ei}/r{repeat}"
                slots.append((key + "/agent", "agent", di, ei, repeat, None))
                for ji, judge in enumerate(manifest["judges"]):
                    if judge["requires_reference"] and example.get("reference") is None:
                        continue
                    slots.append((key + f"/j{ji}", "judge", di, ei, repeat, ji))
    return slots


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def validate_state(state, run_id: str, manifest: dict) -> None:
    require(isinstance(state, dict), "Checkpoint must be a JSON object")
    require(set(state) == {"schema_version", "run_id", "revision", "previous", "manifest", "calls", "published", "timestamp"}, "Invalid checkpoint fields")
    require(state["schema_version"] == SCHEMA_VERSION and type(state["schema_version"]) is int, "Unsupported checkpoint schema")
    require(state["run_id"] == run_id, "Checkpoint run identity mismatch")
    require(canonical(state["manifest"]) == canonical(manifest), "Incompatible resume: dataset, agent, judges, provider configuration or evaluation settings changed")
    require(type(state["revision"]) is int and state["revision"] >= 0, "Invalid checkpoint revision")
    require(state["previous"] is None if state["revision"] == 0 else isinstance(state["previous"], str) and len(state["previous"]) == 64, "Invalid checkpoint predecessor")
    from datetime import datetime
    try:
        timestamp = datetime.fromisoformat(state["timestamp"])
        require(timestamp.tzinfo is not None, "Checkpoint timestamp needs a timezone")
    except (TypeError, ValueError) as exc:
        raise CheckpointError("Invalid checkpoint timestamp") from exc
    calls = state["calls"]
    published = state["published"]
    require(isinstance(calls, dict), "Invalid checkpoint calls")
    require(isinstance(published, list) and len(published) == len(manifest["datasets"]) and all(type(x) is bool for x in published), "Invalid publication state")
    plan = call_plan(manifest)
    require(set(calls) <= {slot[0] for slot in plan}, "Unknown checkpoint call identity")
    unfinished = False
    progress = 0
    for key, kind, di, ei, repeat, ji in plan:
        if key not in calls:
            unfinished = True
            continue
        require(not unfinished, "Invalid call ordering: work exists after an unfinished slot")
        require(all(published[:di]), "Calls precede publication of an earlier dataset")
        call = calls[key]
        require(isinstance(call, dict) and "status" in call, f"Invalid call {key}")
        status = call["status"]
        if status == "pending":
            require(set(call) == {"status"}, f"Invalid pending call {key}")
            unfinished = True
            progress += 1
        elif kind == "agent":
            require(status == "done" and set(call) == {"status", "output", "steps"}, f"Invalid saved agent {key}")
            progress += 2
        else:
            require(status in ("raw", "done") and isinstance(call.get("raw"), str), f"Invalid saved judge {key}")
            expected_keys = {"status", "raw"} if status == "raw" else {"status", "raw", "grade"}
            require(set(call) == expected_keys, f"Invalid judge fields {key}")
            if status == "done":
                judge = restore_judge(manifest["judges"][ji])
                require(canonical(call["grade"]) == canonical(judge.parse_response(call["raw"])), f"Invalid parsed judgement {key}")
            else:
                unfinished = True
            progress += 3 if status == "done" else 2
    for di, completed in enumerate(published):
        if completed:
            require(all(calls.get(slot[0], {}).get("status") == "done" for slot in plan if slot[2] == di), "Publication precedes complete dataset")
            require(all(published[:di]), "Invalid publication ordering")
    require(state["revision"] == progress + sum(published), "Invalid checkpoint progress revision")
    canonical(state)


def validate_successor(old: dict, new: dict, run_id: str, manifest: dict) -> None:
    validate_state(new, run_id, manifest)
    require(new["revision"] == old["revision"] + 1 and new["previous"] == digest(old), "Stale or unrelated checkpoint temp; not recovered")
    require(new["timestamp"] == old["timestamp"], "Checkpoint timestamp changed")
    old_calls, new_calls = old["calls"], new["calls"]
    changed = [key for key in old_calls.keys() | new_calls.keys() if old_calls.get(key) != new_calls.get(key)]
    publications = [i for i, pair in enumerate(zip(old["published"], new["published"])) if pair[0] != pair[1]]
    require(len(changed) + len(publications) == 1, "Invalid checkpoint transition")
    if publications:
        index = publications[0]
        require(not old["published"][index] and new["published"][index], "Publication cannot be undone")
        return
    key = changed[0]
    before, after = old_calls.get(key), new_calls.get(key)
    require(after is not None, "Saved calls cannot be deleted")
    if before is None:
        require(after == {"status": "pending"}, "Call must record pending intent first")
    elif before["status"] == "pending":
        require(after["status"] in ("raw", "done"), "Invalid pending transition")
    else:
        require(before["status"] == "raw" and after["status"] == "done" and before["raw"] == after["raw"], "Saved calls cannot be replaced")


class Checkpoint:
    def __init__(self, path: Path, run_id: str, manifest: dict, initial: dict | None = None, *, validate_artifacts: Callable[[dict], None] | None = None):
        self.path, self.run_id, self.manifest = path, run_id, manifest
        temp = path.with_suffix(path.suffix + ".tmp")
        if initial is not None:
            require(not path.exists() and not temp.exists(), "Run already has checkpoint state")
            validate_state(initial, run_id, manifest)
            if validate_artifacts:
                validate_artifacts(initial)
            atomic_write(path, initial)
            self.state = initial
            return
        if not path.exists() and temp.exists():
            candidate = read_json(temp)
            validate_state(candidate, run_id, manifest)
            require(candidate["revision"] == 0, "Cannot recover noninitial temp without its predecessor")
            if validate_artifacts:
                validate_artifacts(candidate)
            promote_temp(temp, path)
        require(path.exists(), f"Missing checkpoint for run '{run_id}'; do not repeat calls without inspecting run files")
        self.state = read_json(path)
        validate_state(self.state, run_id, manifest)
        if validate_artifacts:
            validate_artifacts(self.state)
        if temp.exists():
            candidate = read_json(temp)
            validate_successor(self.state, candidate, run_id, manifest)
            if validate_artifacts:
                validate_artifacts(candidate)
            promote_temp(temp, path)
            self.state = candidate

    def advance(self, *, calls: dict | None = None, published: list[bool] | None = None) -> None:
        candidate = {
            **self.state, "revision": self.state["revision"] + 1,
            "previous": digest(self.state),
            "calls": self.state["calls"] if calls is None else calls,
            "published": self.state["published"] if published is None else published,
        }
        validate_successor(self.state, candidate, self.run_id, self.manifest)
        atomic_write(self.path, candidate)
        self.state = candidate

    def save_call(self, key: str, call: dict) -> None:
        self.advance(calls={**self.state["calls"], key: call})
