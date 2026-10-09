import json
import os
import selectors
import sys
from pathlib import Path

from evalkit import providers
from evalkit.config import Config
from evalkit.offline import run_offline


def main():
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    cfg = Config(agent="publication_agent:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges")
    counts = root / "calls.json"

    def record(kind):
        value = json.loads(counts.read_text()) if counts.exists() else {"agent": 0, "judge": 0}
        value[kind] += 1
        counts.write_text(json.dumps(value))

    providers.register_provider("mock", lambda model, prompt: (record("judge"), json.dumps({"score": 5, "reasoning": "ok"}))[1])
    if mode == "resume":
        print(json.dumps(run_offline(cfg, "agent-v1", resume="run-1", evaluation_version="eval-v1")), flush=True)
        return

    import evalkit.checkpoint_io as io
    import evalkit.checkpoint_state as checkpoint_state
    import evalkit.resumable as resumable

    boundary = os.environ["PUBLICATION_BOUNDARY"]
    state = {"checkpoint": False}
    original_fsync, original_replace = io.os.fsync, io.os.replace
    original_sync, original_advance, original_publish = io.sync_directory, checkpoint_state.Checkpoint.advance, resumable.publish

    def pause(name):
        if name == boundary:
            print("READY", flush=True)
            with selectors.DefaultSelector() as selector:
                selector.register(sys.stdin, selectors.EVENT_READ)
                if selector.select(30):
                    sys.stdin.readline()

    def fsync(fd):
        original_fsync(fd)
        descriptor = os.fstat(fd)
        temp = Path(".evals/results/offline")
        if temp.exists() and any((item.stat().st_dev, item.stat().st_ino) == (descriptor.st_dev, descriptor.st_ino) for item in temp.glob("*.json.tmp")):
            pause("result-temp-durable")
        checkpoint_temp = Path(".evals/checkpoints/run-1/state.json.tmp")
        if checkpoint_temp.exists() and (checkpoint_temp.stat().st_dev, checkpoint_temp.stat().st_ino) == (descriptor.st_dev, descriptor.st_ino):
            pause("ack-checkpoint-temp" if state["checkpoint"] else "checkpoint-temp-durable")

    def replace(source, target):
        original_replace(source, target)
        name = Path(target).name
        if name.endswith(".json") and "results" in Path(target).parts:
            pause("result-renamed")
        elif name == "state.json":
            pause("ack-checkpoint-renamed" if state["checkpoint"] else "checkpoint-renamed")

    def sync(path):
        original_sync(path)
        if Path(path).name == "offline" and "results" in Path(path).parts:
            pause("result-directory-synced")

    def advance(self, **kwargs):
        state["checkpoint"] = kwargs.get("published") is not None
        try:
            return original_advance(self, **kwargs)
        finally:
            state["checkpoint"] = False

    def publish(checkpoint, index):
        if all(call.get("status") == "done" for key, call in checkpoint.state["calls"].items() if key.startswith(f"d{index}/")):
            pause("all-judge-slots-done")
        result = original_publish(checkpoint, index)
        pause("acknowledged")
        return result

    io.os.fsync = fsync
    io.os.replace = replace
    io.sync_directory = sync
    checkpoint_state.Checkpoint.advance = advance
    resumable.publish = publish
    result = run_offline(cfg, "agent-v1", checkpoint="run-1", evaluation_version="eval-v1")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
