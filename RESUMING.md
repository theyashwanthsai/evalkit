# Resumable offline evaluations

Checkpointing is an opt-in, local JSON prototype. Ordinary `evalkit offline` and online evaluations keep their existing behavior and result schema.

## Start and resume

```sh
evalkit offline --checkpoint experiment-1 --version agent-v3 --evaluation-version config-v2
evalkit offline --resume experiment-1 --version agent-v3 --evaluation-version config-v2
```

Both commands require an explicit agent `--version`. `--evaluation-version` identifies custom evaluation and provider configuration. It can also be supplied in YAML:

```yaml
offline:
  repeats: 2
  checkpoints_dir: .evals/checkpoints
  evaluation_version: config-v2
```

Checkpointing is enabled only by `--checkpoint RUN_ID`; resumption only by `--resume RUN_ID`. These flags are mutually exclusive. Run IDs contain 1-80 letters, digits, underscores or hyphens, starting with a letter or digit. Starting an existing ID fails rather than overwriting it. Run identities are scoped to the checkpoint directory. Keep that directory and the completed-result directory separate and non-overlapping. Use the same working directory and configuration when resuming.

The Python API accepts the same options:

```python
run_offline(cfg, version="agent-v3", checkpoint="experiment-1", evaluation_version="config-v2")
run_offline(cfg, version="agent-v3", resume="experiment-1", evaluation_version="config-v2")
```

## Uncertain calls

Pending intent is saved before every agent or judge call. Agent output and steps are saved before judging. Each judge's raw text is saved before parsing. Compatible saved responses are reused without another call, including raw text whose parsing was interrupted.

If a process stops after pending intent but before saving its response, Evalkit cannot know whether that call happened. Default resume stops before making calls and names the uncertain slot. After inspecting possible side effects and cost, you can explicitly authorize repeating unresolved calls:

```sh
evalkit offline --resume experiment-1 --version agent-v3 --evaluation-version config-v2 --repeat-uncertain
```

This emits a warning about repeated side effects and provider cost. It repeats only unresolved calls, not saved responses. It cannot retrieve a response that never reached disk and does not provide exactly-once external execution. Even pure agents can incur repeated judge cost.

## Compatibility and caller responsibilities

Before calls, resume checks full dataset byte hashes, resolved paths and occurrence order, example order, agent import identity/name and explicit version, full judge definitions/versions, provider/model choices, repeats, regression threshold, result location and evaluation version. Duplicate example IDs and repeated dataset paths have separate occurrence indexes. Judges need unique name/version identities. Reference-required judges remain skipped when a reference is absent.

Judge YAML is fingerprinted in full, but only known evaluation fields are stored, not unrelated YAML keys. Known endpoint, organization and project environment settings are fingerprinted without saving their values. SDK defaults used by Evalkit are included. Credentials are neither read into the manifest nor persisted from configuration. Version labels do not prove that source code or environment is unchanged. Callers must bump the agent version for changes to agent source, dependencies and behavior-affecting configuration, and bump the evaluation version for custom provider implementations, SDK versions or other evaluation settings not described by judge definitions. Custom providers remain responsible for their own behavior and retries.

Checkpointed agent outputs and steps must be JSON-compatible, with no NaN or infinity. Agent inputs, outputs, steps and judge text are stored locally and may themselves contain sensitive data. Do not use fixtures or responses containing credentials. State and lock files are owner-only, and run directories are created owner-only. Do not publish checkpoints.

## Durability and ownership

Each run has a schema-versioned snapshot in `checkpoints_dir/RUN_ID/state.json`. Slot identities include dataset occurrence, example occurrence, repeat, and agent/judge index. State validation checks known identities, call ordering, parsed judgement consistency, progress revision and publication completeness.

Writes serialize valid JSON to a sibling `.tmp`, flush and fsync it, atomically replace the destination, then synchronize the directory where supported. A failed write before replacement leaves the previous valid snapshot. A failure after replacement can leave the new valid snapshot; inspect state before retrying. Resume promotes a complete valid temp only when it matches the run/configuration and is the next linked state transition. An initial revision-zero temp can be recovered without a predecessor. Invalid, truncated, stale or unrelated temps fail safely and are retained for inspection. A corrupt main snapshot is not replaced speculatively.

A nonblocking POSIX `flock` permits one writer per checkpoint-directory/run identity. Another active writer fails. Process death releases ownership automatically; resume reuses the lock file without deleting it or guessing from a PID. Do not delete or replace lock files. This prototype requires a local filesystem with reliable advisory locks, atomic replacement and fsync semantics. Windows and network/distributed filesystems are not supported.

## Completed results

Only complete datasets are published, one deterministic JSON result per run/dataset occurrence. Result content is saved atomically before publication is acknowledged in the checkpoint. Resume verifies already-published files and recovers compatible publication temps without creating another result. Conflicting result files, missing acknowledged results and incomplete publication artifacts stop recovery rather than being overwritten.

Checkpoints live outside completed-result discovery. `.tmp` files are not JSON result files. Reports, comparisons and baseline selection therefore see completed datasets only. A dataset finished before a later dataset was interrupted remains a completed result. Result timestamps identify the run's creation time. Repeating a completed resume returns the same result paths without issuing calls.

Checkpoint errors printed by the CLI exit with status 2. External agent/provider errors and filesystem errors remain failures; preserve the run files, fix the cause, then resume. For malformed state, preserve evidence and inspect files rather than deleting them to force another call.
