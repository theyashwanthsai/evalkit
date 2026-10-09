import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(sys.argv[1])
from evalkit.config import Config
from evalkit.offline import run_offline

run_offline(Config(agent="agentmod:run", datasets=["evals/datasets/*.jsonl"], judges_dir="evals/judges"), "agent-v1", checkpoint="worker", evaluation_version="eval-v1")
