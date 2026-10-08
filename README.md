# evalkit

A small tool I use to eval my own agents. Offline evals before I ship a change, online evals on whatever
real traffic comes in afterwards, and scores kept per git version so I can see if things got better.

It's a dev experiment, not a product. It works for how I build things. Expect rough edges.

## The idea

- **Offline:** run a fixed dataset through the agent, have an LLM judge grade each answer, compare to the
  previous version. Meant to run on every PR.
- **Online:** the deployed agent writes a trace per request. Once a day, pick a handful of recent traces
  and judge them with the same judges (the ones that don't need a reference answer).
- Everything is plain files in the repo: traces, results, judge configs. Versions are `git describe`.

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
evalkit online                  # judge a sample of traces from .traces/
evalkit report                  # scores per version
```

Keys are read from a `.env` in the current directory (or any parent).

### Your agent

Point `evalkit.yaml` at a function that takes the input and returns the output and the steps:

```python
def answer(user_input: str) -> dict:
    ...
    return {"output": "...", "steps": [...]}      # steps can be anything JSON-able

def serve(user_input: str) -> str:                # what production calls
    r = answer(user_input)
    record_trace(user_input, r["output"], steps=r["steps"])
    return r["output"]
```

`answer` is what the offline eval calls. It does not write traces, so eval runs don't end up in
`.traces/`. `serve` is the same agent plus a trace.

### Judges

A judge is a yaml file: prompt, score range, model, version. See `examples/shared/evals/judges/`.
The prompt can use `{input}`, `{output}`, `{reference}` and `{steps}`.

A judge that needs a reference answer is offline only. Production traces don't have one, and evalkit
errors if you enable such a judge for online.

## How comparison works

Scores are compared per example, not just by average. A drop counts as a regression only if it is
bigger than `regression_threshold` and also clear of the noise (roughly a 95% interval). LLM judges
are noisy, so this avoids failing a PR over a coin flip. `offline.repeats` averages several runs.

Each result also stores a hash of the dataset and of every judge. If a judge prompt changed without its
`version` changing, that judge is skipped in the comparison. Otherwise a more lenient judge looks
like an improvement.

## Examples

`examples/` has the same job-search agent (Exa for search) written in LangChain, LangGraph, CrewAI and
PydanticAI. They share one dataset and a few judges. Each folder is just a `my_agent.py` and a
three-line `evalkit.yaml`.

```bash
cd examples/pydanticai
# needs OPENAI_API_KEY and EXA_API_KEY in a .env
evalkit offline
```

## What's rough

- The online side assumes traces are in the repo. Something has to get `.traces/` from your server into
  git (or you run `evalkit online` on the server). I haven't built that part.
- The GitHub workflows from `evalkit init` have not been run in CI yet.
- The Anthropic provider is untested. OpenAI is what I've actually used.
- I tested the four example agents with a stubbed search, not with live Exa results.
- The default judge models in `evalkit init` are placeholders. A weak judge gives noisy scores.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

MIT.
