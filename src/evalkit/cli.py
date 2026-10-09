from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from . import __version__
from .compare import compare, find_baseline, to_markdown
from .config import load_config
from .report import build_report
from .store import load_results
from .version import agent_version


def cmd_init(args) -> int:
    from .templates import FILES
    for rel, content in FILES.items():
        p = Path(rel)
        if p.exists() and not args.force:
            print(f"skip  {rel} (exists)")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        print(f"write {rel}")
    return 0


def cmd_offline(args) -> int:
    from .offline import run_offline
    cfg = load_config(args.config)
    prior = load_results(cfg.results_dir, "offline") if args.compare else []
    md, regressed = [], False
    for result, path in run_offline(cfg, version=args.version, checkpoint=args.checkpoint, resume=args.resume, repeat_uncertain=args.repeat_uncertain, evaluation_version=args.evaluation_version):
        print(f"[{result['dataset']}] {result['agent_version']} -> {path}")
        for j, s in result["summary"].items():
            print(f"  {j}: mean={s['mean']} pass_rate={s['pass_rate']} n={s['n']} unparsed={s['failed_to_parse']}")
        if args.compare:
            base = find_baseline(prior, result, args.baseline)
            if not base:
                md.append(f"### evalkit: `{result['agent_version']}` ({result['dataset']})\n\nNo baseline found, nothing to compare against.")
                continue
            cmp = compare(result, base, cfg.regression_threshold)
            md.append(to_markdown(result, base, cmp))
            regressed |= cmp["regressed"]
    if md:
        text = "\n\n".join(md)
        print("\n" + text)
        if args.markdown:
            Path(args.markdown).write_text(text)
    if regressed and args.fail_on_regression:
        print("\nRegression detected.", file=sys.stderr)
        return 1
    return 0


def cmd_online(args) -> int:
    from .online import run_online
    cfg = load_config(args.config)
    if args.traces_dir:
        cfg.traces_dir = args.traces_dir
    r = run_online(cfg, seed=args.seed)
    if r is None:
        print("No new traces to sample in the window.")
        return 0
    result, path = r
    print(f"sampled {result['sampled']} traces -> {path}")
    for v, summ in result["summary_by_version"].items():
        for j, s in summ.items():
            print(f"  {v} {j}: mean={s['mean']} pass_rate={s['pass_rate']} n={s['n']}")
    return 0


def cmd_report(args) -> int:
    print(build_report(load_config(args.config).results_dir))
    return 0


def cmd_version(args) -> int:
    print(agent_version())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="evalkit", description="Git-native offline + online agent evals.")
    p.add_argument("--config", default="evalkit.yaml")
    p.add_argument("--version-info", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="scaffold config, judges, dataset, workflows")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("offline", help="run datasets through the agent and judge")
    s.add_argument("--version", help="override agent version (default: git describe)")
    runs = s.add_mutually_exclusive_group()
    runs.add_argument("--checkpoint", metavar="RUN_ID", help="start a new checkpointed run")
    runs.add_argument("--resume", metavar="RUN_ID", help="resume an existing checkpointed run")
    s.add_argument("--evaluation-version", help="explicit version of provider/custom evaluation configuration")
    s.add_argument("--repeat-uncertain", action="store_true", help="repeat unresolved pending calls on resume; may duplicate side effects/cost")
    s.add_argument("--compare", action="store_true", help="compare against previous version's result")
    s.add_argument("--baseline", help="baseline agent version (default: latest other version)")
    s.add_argument("--fail-on-regression", action="store_true")
    s.add_argument("--markdown", help="write comparison markdown to this file")
    s.set_defaults(fn=cmd_offline)

    s = sub.add_parser("online", help="judge a sample of production traces")
    s.add_argument("--seed", type=int)
    s.add_argument("--traces-dir", help="read traces from here (e.g. a checkout of your traces repo)")
    s.set_defaults(fn=cmd_online)

    sub.add_parser("report", help="score history per version").set_defaults(fn=cmd_report)
    sub.add_parser("version", help="print resolved agent version").set_defaults(fn=cmd_version)

    load_dotenv(find_dotenv(usecwd=True), override=True)
    args = p.parse_args(argv)
    from .checkpoint_io import CheckpointError
    try:
        return args.fn(args)
    except CheckpointError as exc:
        print(f"Checkpoint error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
