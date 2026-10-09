import json
from pathlib import Path

import pytest

from evalkit import providers
from evalkit.checkpoint_io import CheckpointError, read_json
from evalkit.checkpoint_state import call_plan, digest, validate_state, validate_successor
from evalkit.config import Config
from evalkit.offline import run_offline
from evalkit.report import build_report
from evalkit.resumable import prepare_manifest
from evalkit.store import load_results
from evalkit.compare import compare, find_baseline

JUDGE = """name: quality\nversion: 1\nprovider: mock\nmodel: m\nmodes: [offline]\nscale: [1, 5]\npass_threshold: 4\nprompt: 'Input: {input} Output: {output}'\n"""


def setup(tmp_path, monkeypatch, rows=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    data = rows or [{"id": "e", "input": "q"}]
    (tmp_path / "evals/datasets/d.jsonl").write_text("\n".join(map(json.dumps, data)) + "\n")
    (tmp_path / "agentmod.py").write_text("def run(x): return 'answer-' + str(x)\n")
    cfg = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    providers.register_provider("mock", lambda model, prompt: json.dumps({"score": 5, "reasoning": "ok"}))
    return cfg


def execute(cfg, **kw):
    return run_offline(cfg, "agent-v1", evaluation_version="eval-v1", **kw)


def test_manifest_changes_reject_before_calls(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    original = Path("evals/datasets/d.jsonl").read_bytes()
    second = Path("evals/datasets/second.jsonl")
    cases = [
        ("dataset bytes", lambda c: Path("evals/datasets/d.jsonl").write_bytes(original + b" ")),
        ("order", lambda c: setattr(c, "datasets", ["evals/datasets/second.jsonl", "evals/datasets/d.jsonl"])),
        ("resolved path", lambda c: setattr(c, "datasets", ["evals/datasets/alternate.jsonl"])),
        ("agent spec", lambda c: setattr(c, "agent", "agentmod:other")),
        ("agent name", lambda c: setattr(c, "agent_name", "renamed")),
        ("repeats", lambda c: setattr(c, "repeats", 2)),
        ("regression", lambda c: setattr(c, "regression_threshold", 0.9)),
        ("results path", lambda c: setattr(c, "results_dir", "different-results")),
    ]
    second.write_bytes(original)
    alternate = Path("evals/datasets/alternate.jsonl")
    alternate.write_bytes(original)
    for index, (name, change) in enumerate(cases):
        local = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
        if name != "order": second.unlink(missing_ok=True)
        execute(local, checkpoint=f"r{index}")
        change(local)
        called = []
        providers.register_provider("mock", lambda model, prompt: called.append(prompt) or "{}")
        with pytest.raises(CheckpointError): execute(local, resume=f"r{index}")
        assert called == [], name
        Path("evals/datasets/d.jsonl").write_bytes(original)
    for version, evaluation_version in [("agent-v2", "eval-v1"), ("agent-v1", "eval-v2")]:
        calls = {"agent": 0, "judge": 0}
        import agentmod
        monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
        providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
        with pytest.raises(CheckpointError):
            run_offline(cfg, version, evaluation_version=evaluation_version, resume="r0")
        assert calls == {"agent": 0, "judge": 0}


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":'])
def test_strict_json_rejects_preserving_bytes(tmp_path, raw):
    path = tmp_path / "state.json"
    path.write_text(raw)
    before = path.read_bytes()
    with pytest.raises(CheckpointError): read_json(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("prompt", "changed prompt"), ("version", "2"), ("provider", "other"), ("model", "other-model"), ("scale", "[0, 10]"), ("pass_threshold", "3")])
def test_judge_manifest_fields_reject_before_calls(tmp_path, monkeypatch, field, value):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    judge_path = Path("evals/judges/q.yaml")
    judge_path.write_text(JUDGE.replace(f"{field}: " + ({"prompt": "'Input: {input} Output: {output}'", "version": "1", "provider": "mock", "model": "m", "scale": "[1, 5]", "pass_threshold": "4"}[field]), f"{field}: {value}"))
    calls = []
    providers.register_provider("mock", lambda model, prompt: calls.append(prompt) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == []


@pytest.mark.parametrize("key", ["OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "ANTHROPIC_BASE_URL"])
def test_provider_environment_fingerprint_rejects_before_calls(tmp_path, monkeypatch, key):
    cfg = setup(tmp_path, monkeypatch)
    monkeypatch.setenv(key, "first")
    execute(cfg, checkpoint="r")
    monkeypatch.setenv(key, "second")
    calls = []
    providers.register_provider("mock", lambda model, prompt: calls.append(prompt) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == []


def test_state_identity_revision_order_and_transition_rejected(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    original = read_json(Path(".evals/checkpoints/r/state.json"))
    mutations = [lambda s: s.update(run_id="other"), lambda s: s.update(schema_version=8), lambda s: s.update(revision=-1), lambda s: s.update(previous="bad"), lambda s: s.update(published=[]), lambda s: s.update(calls={"unknown": {"status": "pending"}})]
    for mutate in mutations:
        candidate = json.loads(json.dumps(original)); mutate(candidate)
        with pytest.raises(CheckpointError): validate_state(candidate, "r", original["manifest"])
    bad = json.loads(json.dumps(original)); bad["revision"] += 1
    with pytest.raises(CheckpointError): validate_successor(original, bad, "r", original["manifest"])


def test_invalid_temp_not_promoted_or_called(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch); execute(cfg, checkpoint="r")
    path = Path(".evals/checkpoints/r/state.json"); before = path.read_bytes()
    temp = path.with_suffix(".json.tmp"); temp.write_text('{"bad":true}')
    calls = []; providers.register_provider("mock", lambda model, prompt: calls.append(prompt) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert path.read_bytes() == before and temp.exists() and calls == []


@pytest.mark.parametrize("failure", ["write", "flush", "file_fsync", "rename"])
def test_atomic_write_failure_preserves_snapshot_and_cleans_temp(tmp_path, monkeypatch, failure):
    from evalkit import checkpoint_io as io
    path = tmp_path / "state.json"; io.atomic_write(path, {"revision": 1}); before = path.read_bytes()
    if failure == "rename": monkeypatch.setattr(io.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("rename")))
    else:
        orig = Path.open
        class Wrapped:
            def __init__(self, f): self.f = f
            def __enter__(self): self.f.__enter__(); return self
            def __exit__(self, *a): return self.f.__exit__(*a)
            def __getattr__(self, k): return getattr(self.f, k)
            def write(self, value):
                if failure == "write": raise OSError("write")
                return self.f.write(value)
            def flush(self):
                if failure == "flush": raise OSError("flush")
                return self.f.flush()
        monkeypatch.setattr(Path, "open", lambda self, *a, **k: Wrapped(orig(self, *a, **k)) if self == path.with_suffix(".json.tmp") else orig(self, *a, **k))
        if failure == "file_fsync": monkeypatch.setattr(io.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("file fsync")))
    with pytest.raises(OSError): io.atomic_write(path, {"revision": 2})
    assert path.read_bytes() == before and not path.with_suffix(".json.tmp").exists()


def test_directory_fsync_failure_keeps_valid_replacement(tmp_path, monkeypatch):
    from evalkit import checkpoint_io as io
    path = tmp_path / "state.json"; io.atomic_write(path, {"revision": 1})
    monkeypatch.setattr(io, "sync_directory", lambda p: (_ for _ in ()).throw(OSError("directory fsync")))
    with pytest.raises(OSError): io.atomic_write(path, {"revision": 2})
    assert read_json(path) == {"revision": 2}


def test_same_stem_files_get_distinct_slots_and_results(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    first = Path("evals/datasets/a/same.jsonl"); second = Path("evals/datasets/b/same.jsonl")
    first.parent.mkdir(parents=True); second.parent.mkdir(parents=True)
    data_a = json.dumps({"id":"same","input":"a"}) + "\n"
    data_b = json.dumps({"id":"same","input":"b"}) + "\n"
    first.write_text(data_a); second.write_text(data_b)
    cfg.datasets = ["evals/datasets/a/*.jsonl", "evals/datasets/b/*.jsonl"]
    manifest = prepare_manifest(cfg, "agent-v1", "eval-v1")
    assert len({slot[0] for slot in call_plan(manifest)}) == len(call_plan(manifest))
    execute(cfg, checkpoint="same-stem")
    results = load_results(cfg.results_dir, "offline")
    assert len(results) == 2 and results[0]["dataset_hash"] != results[1]["dataset_hash"]


def test_duplicate_ids_repeated_paths_and_mutating_agent(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [{"id":"same","input":{"q":"a"}}, {"id":"same","input":{"q":"b"}}])
    cfg.datasets = ["evals/datasets/d.jsonl", "evals/datasets/d.jsonl"]
    (tmp_path / "evals/judges/ref.yaml").write_text(JUDGE.replace("name: quality", "name: ref").replace("prompt:", "requires_reference: true\nprompt:"))
    manifest = prepare_manifest(cfg, "agent-v1", "eval-v1")
    assert len({s[0] for s in call_plan(manifest)}) == len(call_plan(manifest))
    import agentmod
    def mutate(value): value["mutated"] = True; return "answer"
    monkeypatch.setattr(agentmod, "run", mutate)
    execute(cfg, checkpoint="r")
    state = read_json(Path(".evals/checkpoints/r/state.json"))
    assert all("mutated" not in row["input"] for ds in state["manifest"]["datasets"] for row in ds["rows"])
    results = load_results(cfg.results_dir, "offline")
    assert len(results) == 2
    assert all({score["judge"] for score in result["scores"]} == {"quality@v1"} for result in results)


@pytest.mark.parametrize("mutation", [
    lambda state: state.update(schema_version=9),
    lambda state: state.update(run_id="wrong"),
    lambda state: state.update(manifest={}),
    lambda state: state.update(previous="0" * 64),
    lambda state: state.update(revision=state["revision"] + 1),
])
def test_invalid_or_stale_checkpoint_temp_rejected_before_calls(tmp_path, monkeypatch, mutation):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    path = Path(".evals/checkpoints/r/state.json")
    before = path.read_bytes()
    candidate = read_json(path)
    mutation(candidate)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(candidate))
    temp_before = temp.read_bytes()
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == {"agent": 0, "judge": 0}
    assert path.read_bytes() == before and temp.read_bytes() == temp_before


def test_saved_output_replacement_rejected_before_calls(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    path = Path(".evals/checkpoints/r/state.json")
    original = read_json(path)
    candidate = json.loads(json.dumps(original))
    agent_key = next(key for key in candidate["calls"] if key.endswith("/agent"))
    candidate["calls"][agent_key]["output"] = "replacement"
    candidate["revision"] += 1
    candidate["previous"] = digest(original)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(candidate))
    before, temp_before = path.read_bytes(), temp.read_bytes()
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == {"agent": 0, "judge": 0}
    assert path.read_bytes() == before and temp.read_bytes() == temp_before


def test_published_output_replacement_rejected_before_calls(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    result = next(Path(".evals/results/offline").glob("*.json"))
    before = result.read_bytes()
    result.write_text('{"replacement":true}')
    replaced = result.read_bytes()
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == {"agent": 0, "judge": 0}
    assert result.read_bytes() == replaced and replaced != before


def test_invalid_publication_temp_preserved_before_resume_calls(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    execute(cfg, checkpoint="r")
    result = next(Path(".evals/results/offline").glob("*.json"))
    temp = result.with_suffix(".json.tmp")
    temp.write_text('{"stale":true}')
    before, temp_before = result.read_bytes(), temp.read_bytes()
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    with pytest.raises(CheckpointError): execute(cfg, resume="r")
    assert calls == {"agent": 0, "judge": 0}
    assert result.read_bytes() == before and temp.read_bytes() == temp_before


@pytest.mark.parametrize("artifact", ["missing", "temp-only", "conflicting-final"])
def test_ack_temp_requires_valid_final_result_before_promotion(tmp_path, monkeypatch, artifact):
    from evalkit import resumable

    cfg = setup(tmp_path, monkeypatch)
    original_publish = resumable.publish
    captured = {}

    def interrupt_before_publication(checkpoint, dataset_index):
        captured["state"] = json.loads(json.dumps(checkpoint.state))
        raise RuntimeError("stop before publication")

    monkeypatch.setattr(resumable, "publish", interrupt_before_publication)
    with pytest.raises(RuntimeError, match="stop before publication"):
        execute(cfg, checkpoint="ack-recovery")
    predecessor = captured["state"]
    assert all(call["status"] == "done" for call in predecessor["calls"].values())
    checkpoint_dir = Path(".evals/checkpoints/ack-recovery")
    state_path = checkpoint_dir / "state.json"
    state_before = state_path.read_bytes()
    monkeypatch.setattr(resumable, "publish", original_publish)
    final = resumable.publication_path(predecessor, 0)
    final.parent.mkdir(parents=True, exist_ok=True)
    result = resumable.checkpoint_result(predecessor, 0)
    if artifact == "temp-only":
        final.with_suffix(final.suffix + ".tmp").write_text(json.dumps(result))
    elif artifact == "conflicting-final":
        final.write_text(json.dumps({**result, "scores": []}))
    candidate = {**predecessor, "revision": predecessor["revision"] + 1,
                 "previous": digest(predecessor), "published": [True]}
    state_temp = state_path.with_suffix(".json.tmp")
    state_temp.write_text(json.dumps(candidate))
    temp_before = state_temp.read_bytes()
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    with pytest.raises(CheckpointError):
        execute(cfg, resume="ack-recovery")
    assert state_path.read_bytes() == state_before
    assert state_temp.read_bytes() == temp_before
    assert calls == {"agent": 0, "judge": 0}


def test_later_dataset_interruption_publishes_and_resumes_only_later_dataset(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch, [{"id": "a", "input": "first"}])
    Path("evals/datasets/b.jsonl").write_text(json.dumps({"id": "b", "input": "second"}) + "\n")
    cfg.datasets = ["evals/datasets/d.jsonl", "evals/datasets/b.jsonl"]
    import agentmod
    counts = {"agent": 0, "judge": 0}
    monkeypatch.setattr(agentmod, "run", lambda x: counts.__setitem__("agent", counts["agent"] + 1) or "answer")
    def judge(model, prompt):
        counts["judge"] += 1
        if counts["judge"] == 2:
            raise RuntimeError("interrupt")
        return json.dumps({"score": 5, "reasoning": "ok"})
    providers.register_provider("mock", judge)
    with pytest.raises(RuntimeError): execute(cfg, checkpoint="multi")
    published = load_results(cfg.results_dir, "offline")
    assert [r["dataset"] for r in published] == ["d"]
    assert "| default | agent-v1 | d |" in build_report(cfg.results_dir)
    assert find_baseline(published, {"dataset": "d", "agent_version": "new", "timestamp": "9999"}) is published[0]
    assert compare(published[0], published[0])["rows"][0]["n"] == 1
    with pytest.raises(CheckpointError): execute(cfg, resume="multi")
    assert counts == {"agent": 2, "judge": 2}
    with pytest.warns(RuntimeWarning, match="Repeating"):
        execute(cfg, resume="multi", repeat_uncertain=True)
    assert counts == {"agent": 2, "judge": 3}
    assert {r["dataset"] for r in load_results(cfg.results_dir, "offline")} == {"d", "b"}


def test_reference_only_judges_skip_calls_but_publish_empty_dataset(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    Path("evals/judges/q.yaml").write_text(JUDGE.replace("modes: [offline]", "modes: [offline]\nrequires_reference: true"))
    calls = {"agent": 0, "judge": 0}
    import agentmod
    monkeypatch.setattr(agentmod, "run", lambda x: calls.__setitem__("agent", calls["agent"] + 1) or "answer")
    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or "{}")
    result, _ = execute(cfg, checkpoint="skip")[0]
    assert calls == {"agent": 1, "judge": 0}
    assert result["scores"] == [] and result["summary"] == {}
    execute(cfg, resume="skip")
    assert calls == {"agent": 1, "judge": 0}


def test_incomplete_run_excluded_from_report_results_and_baselines(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    providers.register_provider("mock", lambda model, prompt: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError): execute(cfg, checkpoint="r")
    assert load_results(cfg.results_dir, "offline") == []
    assert "quality" not in build_report(cfg.results_dir)
    assert find_baseline([], {"agent_version":"current", "dataset":"d"}) is None
