"""Resolve the agent version, so every trace and score is tagged.

Order: EVALKIT_VERSION env -> git describe -> platform commit env vars -> "unknown" (with a warning).
For several agents in one repo, tag them `<agent>/v1.2` and pass the agent name: only tags with
that prefix are considered, so each agent has its own version history.
"""
import os
import subprocess
import sys

# Most hosts expose the deployed commit through one of these when there is no .git folder.
PLATFORM_SHA_VARS = (
    "GITHUB_SHA", "VERCEL_GIT_COMMIT_SHA", "RAILWAY_GIT_COMMIT_SHA", "RENDER_GIT_COMMIT",
    "HEROKU_SLUG_COMMIT", "CF_PAGES_COMMIT_SHA", "COMMIT_SHA", "GIT_COMMIT", "SOURCE_VERSION",
)

_warned = False


def agent_version(agent: str | None = None, cwd: str | None = None) -> str:
    env = os.environ.get("EVALKIT_VERSION")
    if env:
        return env
    cmd = ["git", "describe", "--tags", "--always", "--dirty"]
    if agent and agent != "default":
        cmd += ["--match", f"{agent}/*"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    for var in PLATFORM_SHA_VARS:
        if os.environ.get(var):
            return os.environ[var][:7]
    _warn_once()
    return "unknown"


def _warn_once() -> None:
    global _warned
    if not _warned:
        _warned = True
        print("evalkit: could not determine the agent version (no EVALKIT_VERSION, no git, no known "
              "platform commit var). Traces will be tagged 'unknown' and skipped by online evals. "
              "Set EVALKIT_VERSION at deploy time.", file=sys.stderr)
