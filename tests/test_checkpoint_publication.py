import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit import providers
from evalkit.config import Config
from evalkit.offline import run_offline
from evalkit.store import load_results

JUDGE = """name: quality
version: 1
provider: mock
model: m
modes: [offline]
scale: [1, 5]
pass_threshold: 4
prompt: 'Input: {input} Output: {output}'
"""
BOUNDARIES = ["all-judge-slots-done", "result-temp-durable", "result-renamed", "result-directory-synced", "ack-checkpoint-temp", "ack-checkpoint-renamed", "acknowledged"]


def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    (tmp_path / "evals/datasets/d.jsonl").write_text(json.dumps({"id": "e", "input": "q"}) + "\n")
    calls_path = str(tmp_path / "calls.json")
    (tmp_path / "publication_agent.py").write_text(f"import json, os\ndef run(value):\n p={calls_path!r}; d=json.load(open(p)) if os.path.exists(p) else {{'agent':0,'judge':0}}; d['agent']+=1; json.dump(d,open(p,'w')); return 'answer-'+value\n")
    cfg = Config(agent="publication_agent:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    calls = {"agent": 0, "judge": 0}

    def agent(value):
        calls["agent"] += 1
        return "answer-" + value

    providers.register_provider("mock", lambda model, prompt: calls.__setitem__("judge", calls["judge"] + 1) or json.dumps({"score": 5, "reasoning": "ok"}))
    return cfg, calls


def launch(root, mode, boundary=None):
    worker = Path(__file__).with_name("publication_worker.py")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    if boundary:
        env["PUBLICATION_BOUNDARY"] = boundary
    return subprocess.Popen([sys.executable, str(worker), str(root), mode], cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def read_line(worker):
    with selectors.DefaultSelector() as selector:
        selector.register(worker.stdout, selectors.EVENT_READ)
        assert selector.select(30), "worker handshake timed out"
        return worker.stdout.readline().strip()


def stop(worker):
    try:
        if worker.poll() is None:
            worker.send_signal(signal.SIGKILL)
            assert worker.wait(timeout=5) == -signal.SIGKILL
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream:
                stream.close()


def semantic_result(result):
    return {key: value for key, value in result.items() if key not in {"timestamp", "path"}}


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_sigkill_publication_boundaries_resume_in_fresh_process(tmp_path, monkeypatch, boundary):
    cfg, _ = setup(tmp_path, monkeypatch)
    control = run_offline(cfg, "agent-v1", checkpoint="control", evaluation_version="eval-v1")
    (tmp_path / "calls.json").unlink(missing_ok=True)
    worker = launch(tmp_path, "boundary", boundary)
    try:
        assert read_line(worker) == "READY", worker.stderr.read()
        if boundary == "result-temp-durable":
            evidence = Path(os.environ.get("EVALKIT_EVIDENCE_DIR", tmp_path / "evidence")) / "publication" / tmp_path.name
            shutil.rmtree(evidence, ignore_errors=True)
            evidence.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_path / ".evals/checkpoints/run-1", evidence / "run-1", dirs_exist_ok=True)
            result_dir = tmp_path / ".evals/results/offline"
            (evidence / "offline").mkdir(parents=True, exist_ok=True)
            for temp in result_dir.glob("*-run-1-d0.json.tmp"):
                shutil.copy2(temp, evidence / "offline" / temp.name)
        stop(worker)
    finally:
        stop(worker)
    calls_path = tmp_path / "calls.json"
    before = json.loads(calls_path.read_text()) if calls_path.exists() else {"agent": 0, "judge": 0}
    resumed_worker = launch(tmp_path, "resume")
    try:
        resumed_worker.communicate(timeout=30)
        assert resumed_worker.returncode == 0, resumed_worker.stderr.read()
    finally:
        stop(resumed_worker)
    after = json.loads(calls_path.read_text()) if calls_path.exists() else {"agent": 0, "judge": 0}
    assert after == before
    cfg = Config(agent="publication_agent:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    resumed = run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")
    assert semantic_result(resumed[0][0]) == semantic_result(control[0][0])
    state = json.loads((tmp_path / ".evals/checkpoints/run-1/state.json").read_text())
    assert all(call["status"] == "done" for call in state["calls"].values())
    assert all(state["published"])
    result_path = Path(resumed[0][1])
    content = result_path.read_bytes()
    assert len(load_results(cfg.results_dir, "offline")) == 2
    if boundary == "result-temp-durable":
        evidence = Path(os.environ.get("EVALKIT_EVIDENCE_DIR", tmp_path / "evidence")) / "publication" / tmp_path.name
        saved_state = json.loads((evidence / "run-1/state.json").read_text())
        assert not any(saved_state["published"])
        saved_temps = list((evidence / "offline").glob("*.json.tmp"))
        assert len(saved_temps) == 1
        assert json.loads(saved_temps[0].read_text()) == json.loads(content)
        assert len(list(result_path.parent.glob("checkpoint-*-run-1-d0.json"))) == 1
    run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")
    assert result_path.read_bytes() == content


@pytest.mark.parametrize("contents", ["{", "{}", '{"wrong":true}', '{"kind":"offline","agent":"other"}'])
def test_invalid_or_mismatched_result_temp_is_preserved(tmp_path, monkeypatch, contents):
    cfg, calls = setup(tmp_path, monkeypatch)
    worker = launch(tmp_path, "boundary", "all-judge-slots-done")
    try:
        assert read_line(worker) == "READY"
        state_path = tmp_path / ".evals/checkpoints/run-1/state.json"
        state = json.loads(state_path.read_text())
        state["published"] = [False]
        state_path.write_text(json.dumps(state))
        from evalkit.resumable import publication_path
        from evalkit.checkpoint_state import Checkpoint
        from evalkit.resumable import prepare_manifest
        manifest = prepare_manifest(cfg, "agent-v1", "eval-v1")
        path = publication_path({"manifest": manifest, "run_id": "run-1"}, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(path.suffix + ".tmp").write_text(contents)
        prior_state = state_path.read_bytes()
        stop(worker)
    finally:
        stop(worker)
    with pytest.raises(ValueError):
        run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")
    assert path.with_suffix(path.suffix + ".tmp").read_text() == contents
    assert state_path.read_bytes() == prior_state
    assert not path.exists()
    assert calls == {"agent": 0, "judge": 0}


def test_writer_contention_and_stale_lock_resume_preserve_inode(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    worker = launch(tmp_path, "boundary", "all-judge-slots-done")
    lock = tmp_path / ".evals/checkpoints/run-1.lock"
    try:
        assert read_line(worker) == "READY"
        inode = lock.stat().st_ino
        cfg = Config(agent="publication_agent:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
        before = json.loads((tmp_path / "calls.json").read_text())
        with pytest.raises(ValueError, match="active writer"):
            run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")
        assert json.loads((tmp_path / "calls.json").read_text()) == before
        assert lock.stat().st_ino == inode
        stop(worker)
    finally:
        stop(worker)
    providers.register_provider("mock", lambda model, prompt: (_ for _ in ()).throw(AssertionError("unexpected provider call")))
    run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")
    assert lock.stat().st_ino == inode


def test_completed_resumes_preserve_exact_publication_and_renderer(tmp_path, monkeypatch):
    cfg, calls = setup(tmp_path, monkeypatch)
    expected = run_offline(cfg, "agent-v1", checkpoint="run-1", evaluation_version="eval-v1")
    assert calls["judge"] == 1
    path = Path(expected[0][1])
    content = path.read_bytes()
    for _ in range(2):
        assert run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1") == expected
        assert path.read_bytes() == content
    assert calls == {"agent": 0, "judge": 1}
    from evalkit.report import build_report
    from evalkit.compare import compare
    assert "quality@v1" in build_report(cfg.results_dir)
    assert compare(expected[0][0], expected[0][0])["rows"][0]["delta"] == 0
