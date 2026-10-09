import json
from pathlib import Path

from evalkit import providers
from evalkit.config import Config
from evalkit.offline import run_offline
from evalkit.resumable import prepare_manifest


def test_checkpoint_redacts_credentials_and_hashes_unknown_judge_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals/judges").mkdir(parents=True)
    (tmp_path / "evals/datasets").mkdir(parents=True)
    judge = tmp_path / "evals/judges/q.yaml"
    judge.write_text("""name: quality\nversion: 1\nprovider: mock\nmodel: model-secret\nmodes: [offline]\nscale: [1, 5]\npass_threshold: 4\napi_key: judge-secret-sentinel\nprompt: 'Input: {input} Output: {output}'\n""")
    (tmp_path / "evals/datasets/d.jsonl").write_text(json.dumps({"id": "e", "input": "question"}) + "\n")
    (tmp_path / "agentmod.py").write_text("def run(x): return 'safe output'\n")
    cfg = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    providers.register_provider("mock", lambda model, prompt: json.dumps({"score": 5, "reasoning": "safe"}))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-credential-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-credential-sentinel")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://endpoint-secret.example")
    monkeypatch.setenv("OPENAI_ORG_ID", "organization-secret")
    manifest = prepare_manifest(cfg, "agent-v1", "eval-v1")
    assert "judge-secret-sentinel" not in json.dumps(manifest)
    assert manifest["judges"][0]["definition_hash"]
    first = json.dumps(manifest)
    judge.write_text(judge.read_text().replace("judge-secret-sentinel", "another-unknown-value"))
    changed = prepare_manifest(cfg, "agent-v1", "eval-v1")
    assert first != json.dumps(changed)
    judge.write_text(judge.read_text().replace("another-unknown-value", "judge-secret-sentinel"))
    run_offline(cfg, "agent-v1", evaluation_version="eval-v1", checkpoint="privacy")
    checkpoint = Path(".evals/checkpoints/privacy/state.json")
    raw = checkpoint.read_bytes()
    for secret in (b"openai-credential-sentinel", b"anthropic-credential-sentinel", b"judge-secret-sentinel", b"endpoint-secret.example", b"organization-secret"):
        assert secret not in raw
    state = json.loads(raw)
    result = next(Path(".evals/results/offline").glob("*.json"))
    assert json.loads(result.read_text())["judges"] == {
        "quality@v1": state["manifest"]["judges"][0]["definition_hash"][:10]
    }
    judge.write_text(judge.read_text().replace("judge-secret-sentinel", "changed-sentinel"))
    try:
        run_offline(cfg, "agent-v1", evaluation_version="eval-v1", resume="privacy")
    except Exception as exc:
        from evalkit.checkpoint_io import CheckpointError
        assert isinstance(exc, CheckpointError)
    else:
        raise AssertionError("changed unknown judge field must reject resume")
