import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(sys.argv[1])

from evalkit import providers
from evalkit.checkpoint_io import atomic_write
from evalkit.checkpoint_state import Checkpoint
from evalkit.config import Config
from evalkit.judge import Judge
from evalkit.offline import run_offline

root = Path.cwd()
mode = os.environ.get("FAULT_MODE")
kind = os.environ.get("FAULT_KIND")
counts_path = root / "counts.json"

def increment(key):
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {"agent": 0, "judge": 0}
    counts[key] += 1
    atomic_write(counts_path, counts)

def block(boundary):
    if boundary == mode:
        print(json.dumps({"ready": boundary}), flush=True)
        if sys.stdin.readline(1) != "!":
            raise RuntimeError("parent pipe closed before release")

def agent(value):
    increment("agent")
    block("agent_during")
    return "answer-" + value

(root / "agentmod.py").write_text('import json, os, sys\nfrom pathlib import Path\ndef run(value):\n p=Path("counts.json"); c=json.loads(p.read_text()); c["agent"]+=1; p.write_text(json.dumps(c))\n if os.environ.get("FAULT_MODE")=="agent_during":\n  print(json.dumps({"ready":"agent_during"}), flush=True); sys.stdin.readline(1)\n return "answer-"+value\n')
sys.modules.pop("agentmod", None)

def provider(model, prompt):
    increment("judge")
    block("judge_during")
    return json.dumps({"score": 5, "reasoning": "ok"})
providers.register_provider("mock", provider)

if mode in ("agent_pending", "judge_pending", "agent_returned"):
    original_save = Checkpoint.save_call
    def save_call(self, key, call):
        if kind == "agent" and key.endswith("/agent") and mode == "agent_returned" and call["status"] == "done":
            block(mode)
        original_save(self, key, call)
        if kind == "agent" and key.endswith("/agent") and mode == "agent_pending" and call["status"] == "pending":
            block(mode)
        if kind == "judge" and "/j" in key and mode == "judge_pending" and call["status"] == "pending":
            block(mode)
    Checkpoint.save_call = save_call

if mode == "judge_returned":
    original_request = Judge.request
    def request(self, **kwargs):
        raw = original_request(self, **kwargs)
        block(mode)
        return raw
    Judge.request = request

if mode == "judge_raw":
    original_parse = Judge.parse_response
    def parse_response(self, raw):
        if any(call.get("status") == "raw" for call in json.loads((root / ".evals/checkpoints/run/state.json").read_text())["calls"].values()):
            block(mode)
        return original_parse(self, raw)
    Judge.parse_response = parse_response

if mode in ("io_fsynced", "io_renamed", "initial_temp"):
    import evalkit.checkpoint_io as io
    original_fsync = io.os.fsync
    original_replace = io.os.replace
    def fsync(fd):
        result = original_fsync(fd)
        temp = root / ".evals/checkpoints/run/state.json.tmp"
        descriptor = os.fstat(fd)
        if mode in ("io_fsynced", "initial_temp") and temp.exists() and (temp.stat().st_dev, temp.stat().st_ino) == (descriptor.st_dev, descriptor.st_ino):
            state = json.loads(temp.read_text())
            if mode == "io_fsynced" and json.loads(counts_path.read_text())["agent"]:
                block(mode)
            if mode == "initial_temp" and state["revision"] == 0:
                block(mode)
        return result
    def replace(source, target):
        original_replace(source, target)
        if mode == "io_renamed" and Path(target).name == "state.json" and json.loads(counts_path.read_text())["agent"]:
            block(mode)
    io.os.fsync = fsync
    io.os.replace = replace

cfg = Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
run_id = os.environ.get("RUN_ID", "run")
if mode in ("resume", "resume_default"):
    run_offline(cfg, "agent-v1", resume=run_id, evaluation_version="eval-v1", repeat_uncertain=os.environ.get("REPEAT") == "1")
else:
    run_offline(cfg, "agent-v1", checkpoint=run_id, evaluation_version="eval-v1")
