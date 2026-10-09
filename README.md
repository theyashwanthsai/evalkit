# evalkit

A small tool I use to eval my own agents. Offline evals before I ship a change, online evals on whatever
real traffic comes in afterwards, and scores kept per git version so I can see if things got better.

It's a dev experiment, not a product. It works for how I build things. Expect rough edges.

## The idea

- **Offline:** run a fixed dataset through the agent, have an LLM judge grade each answer, compare to the
  previous version. Meant to run on every PR.
- **Online:** the deployed agent writes a trace per request. Once a day, pick a handful of recent traces
  and judge them with the same judges (the ones that don't need a reference answer).
- Everything is plain files in git: traces, results, judge configs. No service in the middle.

## Install

Not on PyPI. From a clone:

```bash
pip install -e ".[openai]"      # or ".[anthropic]"
```

## Use it

```bash
evalkit init                    # drops in evalkit.yaml, example judges, a dataset, two workflows
evalkit offline                 # run datasets through your agent, judge, save results
evalkit offline --compare       # also compare against the previous version
evalkit online                  # judge a sample of traces
evalkit report                  # scores per agent and version
```

Keys are read from a `.env` in the current directory (or any parent).

### Your agent

Point `evalkit.yaml` at a function that takes the input and returns the output and the steps:

```python
import evalkit
from evalkit import record_trace

def answer(user_input: str) -> dict:              # what `evalkit offline` calls
    ...
    return {"output": "...", "steps": [...]}      # steps can be anything JSON-able

def serve(user_input: str) -> str:                # what production calls
    with evalkit.run():
        r = answer(user_input)
        record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
```

`answer` doesn't write traces, so eval runs don't end up in your production traces. `serve` is the
same agent plus a trace.

### Judges

A judge is a yaml file: prompt, score range, model, version. See `examples/shared/evals/judges/`.
The prompt can use `{input}`, `{output}`, `{reference}` and `{steps}`.

A judge that needs a reference answer is offline only. Production traces don't have one, and evalkit
errors if you enable such a judge for online.

## Traces

Each trace is one JSON object: input, output, steps, error, the agent name and version, plus `run_id`
and `parent_id`. By default they are written to `.traces/<agent>/<date>/<id>.json`.

For production there's a `GitHubSink`. It queues traces and commits them in batches to a **separate
repo** (private, usually) through the GitHub contents API. One `.jsonl` file per agent per day per
flush, no checkout on the server.

```yaml
# evalkit.yaml
agent_name: jobs-agent
traces:
  sink: github
  repo: you/jobs-agent-traces      # needs GITHUB_TOKEN with contents:write on this repo only
  branch: main                     # optional, must already exist
```

Or set `EVALKIT_TRACES_REPO`. A failed flush never touches the request: traces stay queued and retry.
In serverless, call `evalkit.flush()` before returning, background threads get killed.

A separate repo keeps trace commits out of your code history, and the token on the server can't push code.
The daily online workflow checks that repo out and runs `evalkit online --traces-dir traces`.
Traces hold real user input. If the traces repo is public, so is that. There's a `redact` hook for scrubbing
before anything is written:

```python
evalkit.configure(redact=lambda trace: {**trace, "input": scrub(trace["input"])})
```

### Several agents

Give each agent an `agent_name`. Traces, results and versions are all namespaced by it, so one traces
repo can hold many agents. If agents call each other, wrap each `serve` in `evalkit.run()`: they share a
`run_id` and each child records its caller as `parent_id`. Each agent is still judged on its own
input and output.

### Versions

The version on every trace and result is, in order: `EVALKIT_VERSION`, `git describe --tags --always --dirty`,
or a commit env var from the host (`GITHUB_SHA`, `VERCEL_GIT_COMMIT_SHA`, `RAILWAY_GIT_COMMIT_SHA` and a few
more). Most deploys have no `.git` folder, so set one of those at build time if yours doesn't.
If it comes out as `unknown`, evalkit warns, and online evals skip those traces.

In a monorepo with several agents, tag them `jobs-agent/v1.2`. Only tags with the agent's prefix count
for that agent.

## How comparison works

Scores are compared per example, not just by average. A drop counts as a regression only if it is
bigger than `regression_threshold` and also clear of the noise (roughly a 95% interval). LLM judges
are noisy, so this avoids failing a PR over a coin flip. `offline.repeats` averages several runs.

Each result also stores a hash of the dataset and of every judge. If a judge prompt changed without its
`version` changing, that judge is skipped in the comparison. Otherwise a more lenient judge looks
like an improvement.

## Examples

`examples/` has the same job-search agent (Exa for search) written in LangChain, LangGraph, CrewAI and
PydanticAI. They share one dataset and a few judges. Each folder is a `my_agent.py` and a
short `evalkit.yaml`.

```bash
cd examples/pydanticai
# needs OPENAI_API_KEY and EXA_API_KEY in a .env
evalkit offline
```

## What's rough

- The GitHub workflows from `evalkit init` have not been run in CI yet.
- `GitHubSink` is tested against a local fake of the GitHub API, not the real one.
- Failed online traces don't turn into dataset entries yet. That's the next thing I want.
- The Anthropic provider is untested. OpenAI is what I've actually used.
- There's no judge calibration. Nothing tells you whether a judge agrees with a human.

## Resumable offline runs

Opt in with `evalkit offline --checkpoint RUN_ID --version AGENT_VERSION --evaluation-version CONFIG_VERSION`. Resume with `--resume RUN_ID` and the same versions. Uncertain pending calls stop by default. See [Resumable offline evaluations](RESUMING.md) for compatibility, ownership, durability and recovery rules.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

MIT.
