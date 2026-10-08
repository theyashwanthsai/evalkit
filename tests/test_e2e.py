import json
import os
import sys

import pytest

from evalkit import providers, record_trace
from evalkit.cli import main
from evalkit.judge import parse_judgement
from evalkit.traces import load_traces, sample_traces

JUDGE = """name: quality
version: 1
provider: mock
model: m
modes: [offline, online]
scale: [1, 5]
pass_threshold: 4
prompt: |
  in: {input}
  out: {output}
"""
REF_JUDGE = JUDGE.replace("name: quality", "name: refjudge").replace("modes: [offline, online]", "modes: [offline, online]\nrequires_reference: true")


def mock(model, prompt):
    # "good" outputs score 5, "bad" score 2
    return json.dumps({"score": 5 if "good" in prompt.split("out:")[1] else 2, "reasoning": "x"})


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    providers.register_provider("mock", mock)
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/judges/q.yaml").write_text(JUDGE)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    (tmp_path / "evals/datasets/d.jsonl").write_text(
        "\n".join(json.dumps({"id": f"e{i}", "input": f"q{i}"}) for i in range(5)))
    (tmp_path / "evalkit.yaml").write_text("agent: agentmod:run\nonline:\n  sample_size: 5\n")
    (tmp_path / "agentmod.py").write_text("import os\ndef run(x):\n    return os.environ.get('OUT','good') + x\n")
    sys.modules.pop("agentmod", None)
    return tmp_path


def test_parse():
    assert parse_judgement('blah {"score": 4, "reasoning": "ok"}', 1, 5) == (4, "ok")
    assert parse_judgement('{"score": 9}', 1, 5)[0] is None
    assert parse_judgement("nope", 1, 5)[0] is None


def test_offline_regression_detected(proj, monkeypatch):
    monkeypatch.setenv("OUT", "good")
    assert main(["offline", "--version", "v1"]) == 0
    monkeypatch.setenv("OUT", "bad")
    rc = main(["offline", "--version", "v2", "--compare", "--fail-on-regression", "--markdown", "c.md"])
    assert rc == 1
    assert "REGRESSION" in (proj / "c.md").read_text()


def test_offline_no_regression(proj, monkeypatch):
    monkeypatch.setenv("OUT", "good")
    main(["offline", "--version", "v1"])
    assert main(["offline", "--version", "v2", "--compare", "--fail-on-regression"]) == 0


def test_judge_edit_without_bump_is_skipped(proj, monkeypatch):
    monkeypatch.setenv("OUT", "good")
    main(["offline", "--version", "v1"])
    (proj / "evals/judges/q.yaml").write_text(JUDGE + "# tweaked\n")
    monkeypatch.setenv("OUT", "bad")
    assert main(["offline", "--version", "v2", "--compare", "--fail-on-regression", "--markdown", "c.md"]) == 0
    assert "changed without a version bump" in (proj / "c.md").read_text()


def test_online_samples_and_tags_versions(proj, monkeypatch):
    monkeypatch.setenv("EVALKIT_VERSION", "v9")
    for i in range(10):
        record_trace(f"q{i}", "good", error="boom" if i == 0 else None)
    assert len(load_traces()) == 10
    assert main(["online", "--seed", "1"]) == 0
    res = json.loads(next((proj / ".evals/results/online").glob("*.json")).read_text())
    assert res["sampled"] == 5
    assert {s["agent_version"] for s in res["scores"]} == {"v9"}
    assert any(s["errored"] for s in res["scores"])  # error slice always included
    # second run must not re-judge the same traces
    main(["online", "--seed", "2"])
    ids = [s["trace_id"] for f in (proj / ".evals/results/online").glob("*.json") for s in json.loads(f.read_text())["scores"]]
    assert len(ids) == len(set(ids)) == 10


def test_reference_judge_rejected_online(proj):
    (proj / "evals/judges/r.yaml").write_text(REF_JUDGE)
    record_trace("q", "good")
    with pytest.raises(ValueError, match="requires a reference"):
        main(["online"])


def test_init_and_report(proj, capsys):
    os.chdir(proj)
    (proj / "fresh").mkdir()
    os.chdir(proj / "fresh")
    assert main(["init"]) == 0
    assert (proj / "fresh/.github/workflows/evalkit-online.yml").exists()
    assert main(["report"]) == 0
    assert "evalkit report" in capsys.readouterr().out


def test_version_resolution(tmp_path, monkeypatch):
    from evalkit import version
    monkeypatch.chdir(tmp_path)  # no git repo here
    for v in ("EVALKIT_VERSION", *version.PLATFORM_SHA_VARS):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(version, "_warned", False)
    assert version.agent_version() == "unknown"
    assert version._warned
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "abcdef1234567")
    assert version.agent_version() == "abcdef1"
    monkeypatch.setenv("EVALKIT_VERSION", "v3")
    assert version.agent_version() == "v3"


def test_version_path_scoped_tags(tmp_path, monkeypatch):
    import subprocess
    from evalkit import version
    monkeypatch.delenv("EVALKIT_VERSION", raising=False)
    g = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q"); (tmp_path / "f").write_text("1"); g("add", "."); g("commit", "-qm", "1")
    g("tag", "jobs-agent/v1.2"); g("tag", "other-agent/v9")
    assert version.agent_version("jobs-agent", cwd=str(tmp_path)) == "jobs-agent/v1.2"
    assert version.agent_version("other-agent", cwd=str(tmp_path)) == "other-agent/v9"


def test_traces_namespaced_by_agent(proj, monkeypatch):
    monkeypatch.setenv("EVALKIT_VERSION", "v1")
    record_trace("a", "x", agent="jobs")
    record_trace("b", "y", agent="support")
    assert (proj / ".traces/jobs").exists() and (proj / ".traces/support").exists()
    assert [t["input"] for t in load_traces(".traces", "jobs")] == ["a"]
    assert len(load_traces(".traces")) == 2


def test_online_only_reads_its_agent_and_skips_unknown(proj, monkeypatch):
    (proj / "evalkit.yaml").write_text("agent: agentmod:run\nagent_name: jobs\n")
    monkeypatch.setenv("EVALKIT_VERSION", "v1")
    record_trace("a", "good", agent="jobs")
    record_trace("b", "good", agent="support")
    record_trace("c", "good", agent="jobs", version="unknown")
    assert main(["online"]) == 0
    res = json.loads(next((proj / ".evals/results/online").glob("jobs__*.json")).read_text())
    assert res["agent"] == "jobs" and res["sampled"] == 1


def test_run_links_nested_agents(proj, monkeypatch):
    import evalkit
    monkeypatch.setenv("EVALKIT_VERSION", "v1")
    with evalkit.run():
        with evalkit.run():
            child = record_trace("sub", "r", agent="researcher")
        parent = record_trace("q", "a", agent="orchestrator")
    assert child["run_id"] == parent["run_id"] == parent["id"]
    assert child["parent_id"] == parent["id"] and parent["parent_id"] is None
    assert record_trace("solo", "x")["parent_id"] is None


def test_redact_hook(proj, monkeypatch):
    import evalkit
    monkeypatch.setenv("EVALKIT_VERSION", "v1")
    evalkit.configure(redact=lambda t: {**t, "input": "<redacted>"})
    try:
        t = record_trace("my email is a@b.com", "ok")
    finally:
        from evalkit import settings
        settings._state["redact"] = None
    assert t["input"] == "<redacted>"
    assert load_traces()[0]["input"] == "<redacted>"


class FakeGitHub:
    """Tiny local stand-in for the GitHub contents API."""
    def __init__(self, fail=False):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        self.files, self.fail, outer = {}, fail, self

        class H(BaseHTTPRequestHandler):
            def do_PUT(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if outer.fail or self.headers.get("Authorization") != "Bearer tok":
                    self.send_response(401 if not outer.fail else 500); self.end_headers(); self.wfile.write(b"{}"); return
                import base64
                outer.files[self.path] = (base64.b64decode(body["content"]).decode(), body.get("branch"))
                self.send_response(201); self.end_headers(); self.wfile.write(b"{}")
            def log_message(self, *a): pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"


def test_github_sink_batches_per_agent_and_day(proj):
    from evalkit.sinks import GitHubSink
    gh = FakeGitHub()
    sink = GitHubSink("me/traces", branch="traces", token="tok", batch_size=100, api=gh.url)
    ts = "2026-10-08T01:00:00+00:00"
    for i in range(3):
        sink.write({"id": f"a{i}", "agent": "jobs", "timestamp": ts, "input": i})
    sink.write({"id": "b0", "agent": "support", "timestamp": ts, "input": 9})
    assert gh.files == {}            # nothing sent until a flush
    sink.flush()
    assert len(gh.files) == 2        # one file per agent/day, not per trace
    paths = sorted(gh.files)
    assert paths[0].startswith("/repos/me/traces/contents/jobs/2026-10-08/") and paths[0].endswith(".jsonl")
    content, branch = gh.files[paths[0]]
    assert branch == "traces" and len(content.strip().splitlines()) == 3


def test_github_sink_failure_keeps_traces_and_never_raises(proj, capsys):
    from evalkit.sinks import GitHubSink
    gh = FakeGitHub(fail=True)
    sink = GitHubSink("me/traces", token="tok", api=gh.url)
    sink.write({"id": "a", "agent": "jobs", "timestamp": "2026-10-08T01:00:00+00:00"})
    sink.flush()                     # must not raise
    assert "flush failed" in capsys.readouterr().err and len(sink._buf) == 1
    gh.fail = False
    sink.flush()
    assert len(gh.files) == 1 and sink._buf == []


def test_github_sink_needs_token(monkeypatch):
    from evalkit.sinks import GitHubSink
    monkeypatch.delenv("GITHUB_TOKEN", raising=False); monkeypatch.delenv("EVALKIT_GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        GitHubSink("me/traces")


def test_sink_chosen_from_config(proj, monkeypatch):
    from evalkit import settings
    from evalkit.sinks import GitHubSink, LocalSink
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    assert isinstance(settings.get_sink(), LocalSink)
    (proj / "evalkit.yaml").write_text("agent_name: jobs\ntraces:\n  sink: github\n  repo: me/traces\n  branch: main\n")
    s = settings.get_sink()
    assert isinstance(s, GitHubSink) and s.repo == "me/traces" and s.branch == "main"


def test_online_traces_dir_flag(proj, monkeypatch):
    monkeypatch.setenv("EVALKIT_VERSION", "v1")
    record_trace("q", "good")
    import shutil
    shutil.move(".traces", "elsewhere")
    assert main(["online"]) == 0 and not (proj / ".evals/results/online").exists()
    assert main(["online", "--traces-dir", "elsewhere"]) == 0
    assert any((proj / ".evals/results/online").glob("*.json"))
