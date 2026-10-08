"""Resolve the agent version from git, so every trace and score is tagged."""
import os
import subprocess


def agent_version(cwd: str | None = None) -> str:
    """EVALKIT_VERSION env > `git describe --tags --always --dirty` > 'unknown'."""
    env = os.environ.get("EVALKIT_VERSION")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
