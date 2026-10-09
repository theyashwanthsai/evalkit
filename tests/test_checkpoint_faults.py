import json
from pathlib import Path

import pytest

from evalkit import providers
from evalkit.checkpoint_io import CheckpointError, atomic_write, writer_lock
from evalkit.checkpoint_state import Checkpoint
from evalkit.config import Config
from evalkit.offline import run_offline

JUDGE = """name: quality
version: 1
provider: mock
model: m
modes: [offline]
scale: [1, 5]
pass_threshold: 4
prompt: 'Input: {input} Output: {output}'
"""


def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    (tmp_path / "evals/datasets/d.jsonl").write_text(json.dumps({"id": "e", "input": "q"}) + "\n")
    (tmp_path / "agentmod.py").write_text("def run(x): return 'answer-' + x\n")
    cfg = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    providers.register_provider("mock", lambda model, prompt: json.dumps({"score": 5, "reasoning": "ok"}))
    return cfg


def test_live_writer_lock_fails_and_releases_after_owner_exit(tmp_path):
    root = tmp_path / "locks"
    with writer_lock(root, "run"):
        with pytest.raises(CheckpointError, match="active writer"):
            with writer_lock(root, "run"):
                pass
    with writer_lock(root, "run"):
        pass


def test_valid_successor_temp_is_recovered(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    run_offline(cfg, "agent-v1", checkpoint="r", evaluation_version="eval-v1")
    path = Path(".evals/checkpoints/r/state.json")
    state = json.loads(path.read_text())
    key = "d0/e0/r0/agent"
    from evalkit.checkpoint_state import digest
    initial = {
        "schema_version": 1, "run_id": "r", "revision": 0, "previous": None,
        "manifest": state["manifest"], "calls": {}, "published": [False], "timestamp": state["timestamp"],
    }
    from evalkit.checkpoint_io import canonical
    state = {**initial, "revision": 1, "previous": digest(initial), "calls": {key: {"status": "pending"}}}
    path.write_bytes(canonical(initial))
    path.with_suffix(".json.tmp").write_bytes(canonical(state))
    checkpoint = Checkpoint(path, "r", initial["manifest"])
    assert checkpoint.state == state
    assert not path.with_suffix(".json.tmp").exists()


def test_unrelated_temp_never_promoted(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    run_offline(cfg, "agent-v1", checkpoint="r", evaluation_version="eval-v1")
    path = Path(".evals/checkpoints/r/state.json")
    temp = path.with_suffix(".json.tmp")
    temp.write_text('{"schema_version":1}')
    with pytest.raises(CheckpointError):
        run_offline(cfg, "agent-v1", resume="r", evaluation_version="eval-v1")
    assert temp.read_text() == '{"schema_version":1}'
    assert path.exists()


@pytest.mark.parametrize("failure", ["write", "flush", "fsync", "rename"])
def test_atomic_write_fault_preserves_previous_snapshot(tmp_path, monkeypatch, failure):
    path = tmp_path / "state.json"
    atomic_write(path, {"revision": 1})
    if failure == "write":
        monkeypatch.setattr(Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write")))
    elif failure in ("flush", "fsync"):
        monkeypatch.setattr("evalkit.checkpoint_io.os.fsync", lambda fd: (_ for _ in ()).throw(OSError(failure)))
    else:
        monkeypatch.setattr("evalkit.checkpoint_io.os.replace", lambda *args: (_ for _ in ()).throw(OSError("rename")))
    with pytest.raises(OSError):
        atomic_write(path, {"revision": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"revision": 1}
