#!/usr/bin/env python3
"""Run codeloop over a SWE-bench subset and emit predictions + per-run metrics.

Each instance gets its own container built from the official SWE-bench image, so
instances are independent and the run is safe to parallelise and to resume.

    python eval/run_swebench.py --n 30 --workers 4 --run-name sr-baseline

Produces eval/results/<run-name>/
    predictions.json      SWE-bench submission format
    metrics.json          aggregate token/step/edit-failure stats
    trajectories/*.json   full message history per instance
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeloop.agent import Agent  # noqa: E402
from codeloop.env import DockerEnvironment  # noqa: E402

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}

TASK_TEMPLATE = """Solve the following issue in the repository at {cwd}.

The repository is already checked out at the relevant commit. Do not create a
branch, do not commit, and do not modify any test file -- your change is graded
by running the project's existing tests against your edit.

<issue>
{problem_statement}
</issue>

Locate the cause, make the minimal fix, and verify it before you finish."""

_LOCK = threading.Lock()


def image_for(instance: dict) -> str:
    # Docker forbids double underscores in tags, so SWE-bench substitutes a token.
    iid = instance["instance_id"].replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()


def run_instance(instance: dict, args, out_dir: Path) -> dict:
    iid = instance["instance_id"]
    env = None
    try:
        env = DockerEnvironment(image_for(instance), cwd="/testbed", platform=args.platform)
        env.execute("git config --global --add safe.directory /testbed")

        agent = Agent(
            env=env,
            model=args.model,
            edit_format=args.edit_format,
            max_steps=args.max_steps,
        )
        agent.run(
            TASK_TEMPLATE.format(cwd=env.cwd, problem_statement=instance["problem_statement"])
        )

        # Stage everything so new files land in the diff too, then read the patch
        # back out. Tests are excluded: SWE-bench grades with its own test files.
        env.execute("git add -A")
        patch = env.execute("git diff --cached")["output"]

        record = {
            "instance_id": iid,
            "patch": patch,
            "usage": agent.trajectory()["usage"],
            "exit_reason": agent.exit_reason,
            "error": None,
        }
        agent.dump_trajectory(str(out_dir / "trajectories" / f"{iid}.json"), instance_id=iid)
        return record
    except Exception:
        return {
            "instance_id": iid,
            "patch": "",
            "usage": {},
            "exit_reason": "error",
            "error": traceback.format_exc(limit=3),
        }
    finally:
        if env is not None:
            env.cleanup()


def write_prediction(out_dir: Path, record: dict, model_name: str) -> None:
    path = out_dir / "predictions.json"
    with _LOCK:
        data = json.loads(path.read_text()) if path.exists() else {}
        data[record["instance_id"]] = {
            "instance_id": record["instance_id"],
            "model_name_or_path": model_name,
            "model_patch": record["patch"],
        }
        path.write_text(json.dumps(data, indent=2))


def summarise(records: list, args) -> dict:
    def col(key):
        return [r["usage"].get(key, 0) for r in records if r["usage"]]

    attempts = sum(col("edit_attempts"))
    failures = sum(col("edit_failures"))
    steps = col("steps")
    total_in, total_out = sum(col("input_tokens")), sum(col("output_tokens"))
    return {
        "run_name": args.run_name,
        "model": args.model,
        "edit_format": args.edit_format,
        "n_instances": len(records),
        "n_empty_patch": sum(1 for r in records if not r["patch"].strip()),
        "n_error": sum(1 for r in records if r["exit_reason"] == "error"),
        "n_step_limit": sum(1 for r in records if r["exit_reason"] == "step_limit"),
        "mean_steps": round(statistics.mean(steps), 1) if steps else 0,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "mean_tokens_per_instance": round((total_in + total_out) / len(records)) if records else 0,
        "total_cache_read_tokens": sum(col("cache_read_tokens")),
        "edit_attempts": attempts,
        "edit_failures": failures,
        "edit_failure_rate": round(failures / attempts, 4) if attempts else 0.0,
        "note": "resolve rate is NOT computed here -- run eval/score.py on predictions.json",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=list(DATASETS), default="verified")
    p.add_argument("--n", type=int, default=30, help="subset size (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="subset sampling seed")
    p.add_argument("--instances", nargs="*", help="explicit instance ids, overrides --n/--seed")
    p.add_argument("-m", "--model", default="claude-sonnet-5")
    p.add_argument("--edit-format", default="search_replace", choices=["search_replace", "whole_file"])
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--platform", default="linux/amd64")
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", action="store_true", help="skip instances already in predictions.json")
    args = p.parse_args()

    from datasets import load_dataset  # imported late: heavy, and only needed here

    ds = load_dataset(DATASETS[args.split], split="test")
    instances = list(ds)
    if args.instances:
        wanted = set(args.instances)
        instances = [i for i in instances if i["instance_id"] in wanted]
    elif args.n:
        # Sort first so the sample depends only on the seed, not on dataset order.
        instances.sort(key=lambda i: i["instance_id"])
        instances = random.Random(args.seed).sample(instances, min(args.n, len(instances)))

    out_dir = Path(__file__).parent / "results" / args.run_name
    (out_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    if args.resume and (out_dir / "predictions.json").exists():
        done = set(json.loads((out_dir / "predictions.json").read_text()))
        before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in done]
        print(f"resuming: {before - len(instances)} already done, {len(instances)} to go")

    print(f"[{args.run_name}] {len(instances)} instances · {args.edit_format} · {args.workers} workers")

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_instance, i, args, out_dir): i["instance_id"] for i in instances}
        for n, fut in enumerate(as_completed(futures), 1):
            record = fut.result()
            records.append(record)
            write_prediction(out_dir, record, f"codeloop-{args.edit_format}")
            mark = "!" if record["exit_reason"] == "error" else ("." if record["patch"].strip() else "0")
            print(f"  [{n}/{len(instances)}] {mark} {record['instance_id']} ({record['exit_reason']})")

    metrics = summarise(records, args)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
