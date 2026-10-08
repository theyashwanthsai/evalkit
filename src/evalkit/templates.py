CONFIG = """agent: my_agent:run          # module:function, takes input, returns str or {"output":..., "steps":[...]}
traces_dir: .traces
results_dir: .evals/results
datasets: ["evals/datasets/*.jsonl"]
judges_dir: evals/judges
offline:
  repeats: 1                 # raise to 3+ to average out judge noise
  regression_threshold: 0.2  # min mean drop (in score points) that counts as a regression
online:
  sample_size: 20
  window_hours: 24
  include_errors: 3          # always include up to N errored traces in the sample
"""

JUDGE_HELPFUL = """name: helpfulness
version: 1
provider: anthropic
model: claude-haiku-4-5-20251001
modes: [offline, online]     # reference-free, so it works on production traces
requires_reference: false
scale: [1, 5]
pass_threshold: 4
prompt: |
  You are grading an AI agent's response.

  User input:
  {input}

  Agent output:
  {output}

  Score 1-5 for how helpful, correct and complete the response is.
"""

JUDGE_CORRECT = """name: correctness
version: 1
provider: anthropic
model: claude-haiku-4-5-20251001
modes: [offline]             # needs a reference answer, so offline only
requires_reference: true
scale: [1, 5]
pass_threshold: 4
prompt: |
  User input:
  {input}

  Reference answer:
  {reference}

  Agent output:
  {output}

  Score 1-5 for whether the agent output is factually consistent with the reference.
"""

DATASET = """{"id": "greet", "input": "Say hello", "reference": "A friendly greeting"}
{"id": "math", "input": "What is 2+2?", "reference": "4"}
"""

AGENT = '''from evalkit import record_trace


def run(user_input: str) -> str:
    output = f"echo: {user_input}"  # replace with your agent
    record_trace(user_input, output)  # writes .traces/<date>/<id>.json tagged with the git version
    return output
'''

WF_OFFLINE = """name: evalkit-offline
on:
  pull_request:
  push:
    branches: [main]
permissions:
  contents: write
  pull-requests: write
jobs:
  offline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }   # needed for `git describe` versions
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install "evalkit[anthropic]" -r requirements.txt
      - name: Run offline evals (compare against last main result)
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: evalkit offline --compare --fail-on-regression --markdown evalkit-comment.md
      - name: Comment on PR
        if: github.event_name == 'pull_request' && always()
        uses: marocchino/sticky-pull-request-comment@v2
        with: { path: evalkit-comment.md }
      - name: Commit result on main
        if: github.ref == 'refs/heads/main'
        run: |
          git config user.name "evalkit[bot]"; git config user.email "evalkit@users.noreply.github.com"
          git add .evals/results && git commit -m "evalkit: offline results" || true
          git push || true
"""

WF_ONLINE = """name: evalkit-online
on:
  schedule: [{ cron: "0 6 * * *" }]   # daily
  workflow_dispatch:
permissions:
  contents: write
jobs:
  online:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install "evalkit[anthropic]"
      - name: Judge a sample of production traces
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: evalkit online
      - name: Commit results
        run: |
          git config user.name "evalkit[bot]"; git config user.email "evalkit@users.noreply.github.com"
          git add .evals/results && git commit -m "evalkit: online results" || true
          git push || true
"""

FILES = {
    "evalkit.yaml": CONFIG,
    "evals/judges/helpfulness.yaml": JUDGE_HELPFUL,
    "evals/judges/correctness.yaml": JUDGE_CORRECT,
    "evals/datasets/smoke.jsonl": DATASET,
    "my_agent.py": AGENT,
    ".github/workflows/evalkit-offline.yml": WF_OFFLINE,
    ".github/workflows/evalkit-online.yml": WF_ONLINE,
}
