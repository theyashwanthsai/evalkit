import json
import os
import shutil
import signal
import subprocess
import selectors
import sys
from pathlib import Path

import pytest

from evalkit.config import Config

EVIDENCE_ROOT = Path(os.environ.get("EVALKIT_EVIDENCE_DIR", "/tmp/evalkit-resumable-evidence")) / "process"
JUDGE = """name: quality
version: 1
provider: mock
model: m
modes: [offline]
scale: [1, 5]
pass_threshold: 4
prompt: 'Input: {input} Output: {output}'
"""
PYTHON = sys.executable


def worker(project, mode=None, kind=None, repeat=False):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    for key, value in (("FAULT_MODE", mode), ("FAULT_KIND", kind), ("REPEAT", "1" if repeat else None), ("RUN_ID", "control" if mode == "control" else None)):
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.Popen(
        [PYTHON, str(Path(__file__).with_name("resume_process_worker.py")), str(project)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        bufsize=1, env=env,
    )


def ready_child(project, mode, kind=None):
    child = worker(project, mode, kind)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=8), "child did not reach fault boundary"
        line = child.stdout.readline()
        assert line, "child exited before fault boundary"
        assert json.loads(line) == {"ready": mode}
        os.kill(child.pid, signal.SIGKILL)
        assert child.wait(timeout=8) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=8)
        child.stdout.close()
        child.stderr.close()
        child.stdin.close()


def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets/d.jsonl").write_text(json.dumps({"id": "e", "input": "q"}) + "\n")
    (tmp_path / "counts.json").write_text('{"agent":0,"judge":0}')
    return Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")


def saved(project_path):
    path = project_path / ".evals/checkpoints/run/state.json"
    state = json.loads(path.read_text())
    target = EVIDENCE_ROOT / project_path.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target / "state.json")
    shutil.copy2(project_path / "counts.json", target / "counts.json")
    return state


@pytest.mark.parametrize(("mode", "kind", "status", "counts"), [
    ("agent_pending", "agent", "pending", {"agent": 0, "judge": 0}),
    ("agent_during", "agent", "pending", {"agent": 1, "judge": 0}),
    ("agent_returned", "agent", "pending", {"agent": 1, "judge": 0}),
    ("judge_pending", "judge", "pending", {"agent": 1, "judge": 0}),
    ("judge_during", "judge", "pending", {"agent": 1, "judge": 1}),
    ("judge_returned", "judge", "pending", {"agent": 1, "judge": 1}),
])
def test_process_fault_matrix(tmp_path, monkeypatch, mode, kind, status, counts):
    cfg = project(tmp_path, monkeypatch)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ready_child(tmp_path, mode, kind)
    state = saved(tmp_path)
    slot = f"d0/e0/r0/{'agent' if kind == 'agent' else 'j0'}"
    assert state["calls"][slot]["status"] == status
    assert json.loads((tmp_path / "counts.json").read_text()) == counts
    if status == "pending":
        rejected = worker(tmp_path, "resume_default")
        assert rejected.wait(timeout=8) != 0
        assert "Uncertain pending call" in rejected.stderr.read()
        rejected.stdout.close(); rejected.stderr.close(); rejected.stdin.close()
        assert json.loads((tmp_path / "counts.json").read_text()) == counts
        repeated = worker(tmp_path, "resume", kind, repeat=True)
    else:
        repeated = worker(tmp_path, "resume")
    assert repeated.wait(timeout=12) == 0, repeated.stderr.read()
    resume_stderr = repeated.stderr.read()
    repeated.stdout.close(); repeated.stderr.close(); repeated.stdin.close()
    expected_counts = {
        "agent": 2 if mode in ("agent_during", "agent_returned") else 1,
        "judge": 2 if mode in ("judge_during", "judge_returned") else 1,
    }
    assert json.loads((tmp_path / "counts.json").read_text()) == expected_counts
    final_state = json.loads((tmp_path / ".evals/checkpoints/run/state.json").read_text())
    assert all(call["status"] == "done" for call in final_state["calls"].values())
    assert len(final_state["calls"]) == 2
    assert final_state["published"] == [True]
    results = list((tmp_path / ".evals/results/offline").glob("checkpoint-*-run-d0.json"))
    assert len(results) == 1
    assert len(json.loads(results[0].read_text())["scores"]) == 1
    if mode in ("agent_during", "agent_returned", "judge_during"):
        assert "possible repeated side effects and provider cost" in resume_stderr


@pytest.mark.parametrize("mode", ["judge_raw", "io_fsynced", "io_renamed"])
def test_durable_temp_recovers_in_fresh_process(tmp_path, monkeypatch, mode):
    cfg = project(tmp_path, monkeypatch)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ready_child(tmp_path, mode)
    before = saved(tmp_path)
    expected_before = {"agent": 1, "judge": 1} if mode == "judge_raw" else {"agent": 1, "judge": 0}
    assert json.loads((tmp_path / "counts.json").read_text()) == expected_before
    control_path = tmp_path / "control"
    control_path.mkdir()
    (control_path / "evals").symlink_to(tmp_path / "evals", target_is_directory=True)
    (control_path / "agentmod.py").symlink_to(tmp_path / "agentmod.py")
    (control_path / "counts.json").write_text('{"agent":0,"judge":0}')
    child = worker(control_path, "control")
    try:
        assert child.wait(timeout=12) == 0, child.stderr.read()
    finally:
        child.stdout.close(); child.stderr.close(); child.stdin.close()
    control = json.loads(next((control_path / ".evals/results/offline").glob("checkpoint-*-control-d0.json")).read_text())
    child = worker(tmp_path, "resume")
    try:
        assert child.wait(timeout=12) == 0, child.stderr.read()
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait(timeout=8)
        child.stdout.close(); child.stderr.close(); child.stdin.close()
    after = saved(tmp_path)
    result_files = list((tmp_path / ".evals/results/offline").glob("checkpoint-*-run-d0.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text())["scores"] == control["scores"]
    assert json.loads((tmp_path / "counts.json").read_text()) == {"agent": 1, "judge": 1}
    assert after["calls"] and before["calls"]


def test_initial_revision_zero_temp_recovery(tmp_path, monkeypatch):
    cfg = project(tmp_path, monkeypatch)
    state_path = tmp_path / ".evals/checkpoints/run/state.json"
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    ready_child(tmp_path, "initial_temp")
    assert not state_path.exists()
    temp = state_path.with_suffix(".json.tmp")
    assert json.loads(temp.read_text())["revision"] == 0
    assert json.loads((tmp_path / "counts.json").read_text()) == {"agent": 0, "judge": 0}
    target = EVIDENCE_ROOT / tmp_path.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temp, target / "initial-state.json.tmp")
    resume_child = worker(tmp_path, "resume")
    try:
        assert resume_child.wait(timeout=12) == 0, resume_child.stderr.read()
    finally:
        if resume_child.poll() is None:
            os.kill(resume_child.pid, signal.SIGKILL)
            resume_child.wait(timeout=8)
        resume_child.stdout.close(); resume_child.stderr.close(); resume_child.stdin.close()
    assert json.loads((tmp_path / "counts.json").read_text()) == {"agent": 1, "judge": 1}
    final_state = json.loads(state_path.read_text())
    assert final_state["published"] == [True]
    assert all(call["status"] == "done" for call in final_state["calls"].values())
    results = list((tmp_path / ".evals/results/offline").glob("checkpoint-*-run-d0.json"))
    assert len(results) == 1 and len(json.loads(results[0].read_text())["scores"]) == 1
