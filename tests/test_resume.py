import json
import os
import signal
import subprocess
import sys
import selectors
from pathlib import Path

import pytest

from evalkit import providers
from evalkit.config import Config
from evalkit.offline import run_offline
from evalkit.judge import Judge
from evalkit.report import load_results

JUDGE = """name: quality
version: 1
provider: mock
model: m
modes: [offline]
scale: [1, 5]
pass_threshold: 4
prompt: 'Input: {input} Output: {output}'
"""


def mock_provider(model, prompt):
    return json.dumps({"score": 5, "reasoning": "ok"})


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    providers.register_provider("mock", mock_provider)
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    (tmp_path / "evals/datasets/d.jsonl").write_text(
        '\n'.join(json.dumps({"id": f"e{i}", "input": f"q{i}"}) for i in range(3)) + '\n'
    )
    (tmp_path / "agentmod.py").write_text("def run(x): return 'answer-' + x\n")
    sys.modules.pop("agentmod", None)
    cfg = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    return cfg


def execute(cfg, run_id="r1", **kwargs):
    return run_offline(cfg, "agent-v1", checkpoint=run_id, evaluation_version="eval-v1", **kwargs)


def test_completed_resume_reuses_agent_and_judge(setup_run, monkeypatch):
    cfg = setup_run
    counts = {"agent": 0, "judge": 0}
    import agentmod
    original = agentmod.run
    def agent(value):
        counts["agent"] += 1
        return original(value)
    monkeypatch.setattr(agentmod, "run", agent)
    prompts = []
    def provider(model, prompt):
        counts["judge"] += 1
        prompts.append(prompt)
        return mock_provider(model, prompt)
    providers.register_provider("mock", provider)
    first = execute(cfg)
    assert counts == {"agent": 3, "judge": 3}
    assert all(f"Input: q{i} Output: answer-q{i}" in prompt for i, prompt in enumerate(prompts))
    assert all('score": <integer between 1 and 5>' in prompt for prompt in prompts)
    resumed = run_offline(cfg, "agent-v1", resume="r1", evaluation_version="eval-v1")
    assert counts == {"agent": 3, "judge": 3}
    assert resumed == first
    assert Path(first[0][1]).exists()


def test_repeats_judges_duplicate_ids_and_reference_skips(setup_run, monkeypatch):
    cfg = setup_run
    cfg.repeats = 2
    (Path("evals/datasets/d.jsonl")).write_text(
        json.dumps({"id": "same", "input": "without-reference"}) + "\n" +
        json.dumps({"id": "same", "input": "with-reference", "reference": "target"}) + "\n"
    )
    (Path("evals/judges/ref.yaml")).write_text(JUDGE.replace("name: quality", "name: reference").replace("modes: [offline]", "modes: [offline]\nrequires_reference: true"))
    counts = {"agent": 0, "judge": 0}
    import agentmod
    original = agentmod.run
    def agent(value):
        counts["agent"] += 1
        return original(value)
    monkeypatch.setattr(agentmod, "run", agent)
    providers.register_provider("mock", lambda model, prompt: counts.__setitem__("judge", counts["judge"] + 1) or mock_provider(model, prompt))
    execute(cfg)
    assert counts == {"agent": 4, "judge": 6}
    state = json.loads(Path(".evals/checkpoints/r1/state.json").read_text())
    assert len(state["calls"]) == 10
    assert len({key for key in state["calls"]}) == 10


def test_incompatible_manifest_rejected_before_calls(setup_run, monkeypatch):
    cfg = setup_run
    execute(cfg)
    called = []
    providers.register_provider("mock", lambda model, prompt: called.append(prompt) or mock_provider(model, prompt))
    (Path("evals/datasets/d.jsonl")).write_text(json.dumps({"id": "changed", "input": "x"}) + '\n')
    with pytest.raises(ValueError, match="Incompatible resume"):
        run_offline(cfg, "agent-v1", resume="r1", evaluation_version="eval-v1")
    assert called == []


def test_corrupt_checkpoint_rejected_without_provider_calls(setup_run, monkeypatch):
    cfg = setup_run
    execute(cfg)
    Path(".evals/checkpoints/r1/state.json").write_text('{"truncated":')
    called = []
    providers.register_provider("mock", lambda model, prompt: called.append(prompt) or mock_provider(model, prompt))
    with pytest.raises(ValueError):
        run_offline(cfg, "agent-v1", resume="r1", evaluation_version="eval-v1")
    assert called == []


def test_saved_raw_judge_response_resumes_parsing_without_provider_call(setup_run, monkeypatch):
    cfg = setup_run
    calls = []
    providers.register_provider("mock", lambda model, prompt: calls.append(prompt) or mock_provider(model, prompt))
    original = Judge.parse_response
    def interrupt(self, raw):
        raise RuntimeError("parse interrupted")
    monkeypatch.setattr(Judge, "parse_response", interrupt)
    with pytest.raises(RuntimeError, match="parse interrupted"):
        execute(cfg)
    state = json.loads(Path(".evals/checkpoints/r1/state.json").read_text())
    assert next(iter(state["calls"].values()))["status"] == "done"
    judge_key = next(key for key, value in state["calls"].items() if value["status"] == "raw")
    first_prompt = calls[0]
    monkeypatch.setattr(Judge, "parse_response", original)
    run_offline(cfg, "agent-v1", resume="r1", evaluation_version="eval-v1")
    assert calls.count(first_prompt) == 1
    assert json.loads(Path(".evals/checkpoints/r1/state.json").read_text())["calls"][judge_key]["status"] == "done"


def test_incomplete_checkpoint_is_excluded_from_results(setup_run):
    cfg = setup_run
    providers.register_provider("mock", lambda model, prompt: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        execute(cfg)
    assert load_results(cfg.results_dir, "offline") == []


def test_resume_rejects_uncertain_call_and_opt_in_warns(setup_run):
    cfg = setup_run
    calls = []
    agent_calls = []
    import agentmod
    original = agentmod.run
    def agent(value):
        agent_calls.append(value)
        return original(value)
    agentmod.run = agent
    def interrupted(model, prompt):
        calls.append(prompt)
        raise RuntimeError("provider interrupted")
    providers.register_provider("mock", interrupted)
    with pytest.raises(RuntimeError, match="provider interrupted"):
        execute(cfg)
    assert len(calls) == 1
    assert agent_calls == ["q0"]
    providers.register_provider("mock", mock_provider)
    with pytest.raises(ValueError, match="Uncertain pending call"):
        run_offline(cfg, "agent-v1", resume="r1", evaluation_version="eval-v1")
    with pytest.warns(RuntimeWarning, match="Repeating"):
        run_offline(cfg, "agent-v1", resume="r1", repeat_uncertain=True, evaluation_version="eval-v1")
    assert agent_calls.count("q0") == 1


def test_kill_only_started_worker_at_agent_call_boundary(setup_run):
    cfg = setup_run
    Path("agentmod.py").write_text(
        "import sys\ndef run(value):\n print('READY', flush=True)\n sys.stdin.readline()\n return value\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
    worker = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("resume_worker.py")), str(Path.cwd())],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(worker.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=8), "worker READY handshake timed out"
        assert worker.stdout.readline().strip() == "READY"
        worker.send_signal(signal.SIGKILL)
        assert worker.wait(timeout=5) == -signal.SIGKILL
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        worker.stdout.close()
        worker.stdin.close()
        if worker.stderr:
            worker.stderr.close()
    state = json.loads(Path(".evals/checkpoints/worker/state.json").read_text())
    assert next(iter(state["calls"].values())) == {"status": "pending"}
    with pytest.raises(ValueError, match="Uncertain pending call"):
        run_offline(cfg, "agent-v1", resume="worker", evaluation_version="eval-v1")
