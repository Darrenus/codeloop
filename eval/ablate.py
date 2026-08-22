#!/usr/bin/env python3
"""Run the same SWE-bench subset under two or more configurations and print a
comparison table.

The point of the whole repo. Two arms differ in exactly one variable and are
handed an identical instance list, so the delta in tokens, edit-failure rate and
resolve rate is attributable.

    python eval/ablate.py --n 30 --arms search_replace whole_file
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}

ROWS = [
    ("Resolve rate", lambda m: f"{m['resolve_rate']:.1%}" if "resolve_rate" in m else "not graded"),
    ("Empty patches", lambda m: str(m.get("n_empty_patch", "-"))),
    ("Mean steps", lambda m: str(m.get("mean_steps", "-"))),
    ("Mean tokens / instance", lambda m: f"{m.get('mean_tokens_per_instance', 0):,}"),
    ("Output tokens (total)", lambda m: f"{m.get('total_output_tokens', 0):,}"),
    ("Edit attempts", lambda m: str(m.get("edit_attempts", "-"))),
    ("Edit failure rate", lambda m: f"{m.get('edit_failure_rate', 0):.1%}"),
    ("Hit step limit", lambda m: str(m.get("n_step_limit", "-"))),
    ("Errored", lambda m: str(m.get("n_error", "-"))),
]


def pick_instances(split: str, n: int, seed: int) -> list:
    from datasets import load_dataset

    ids = sorted(i["instance_id"] for i in load_dataset(DATASETS[split], split="test"))
    return random.Random(seed).sample(ids, min(n, len(ids)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["search_replace", "whole_file"])
    p.add_argument("--split", choices=list(DATASETS), default="verified")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-p", "--provider", default="anthropic")
    p.add_argument("-m", "--model", default=None)
    p.add_argument("--cache-dir", help="record and replay completions (free reruns)")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="ablation")
    p.add_argument("--skip-grading", action="store_true")
    args = p.parse_args()

    instances = pick_instances(args.split, args.n, args.seed)
    print(f"{len(instances)} instances, seed {args.seed}, arms: {', '.join(args.arms)}\n")

    metrics = {}
    for arm in args.arms:
        run_name = f"{args.tag}-{arm}"
        subprocess.run(
            [sys.executable, str(HERE / "run_swebench.py"),
             "--split", args.split, "--instances", *instances,
             "--provider", args.provider, "--edit-format", arm,
             *(["--model", args.model] if args.model else []),
             *(["--cache-dir", args.cache_dir] if args.cache_dir else []),
             "--max-steps", str(args.max_steps), "--workers", str(args.workers),
             "--run-name", run_name, "--resume"],
            check=True,
        )
        if not args.skip_grading:
            subprocess.run(
                [sys.executable, str(HERE / "score.py"),
                 "--run-name", run_name, "--split", args.split,
                 "--workers", str(args.workers)],
                check=False,
            )
        metrics[arm] = json.loads((HERE / "results" / run_name / "metrics.json").read_text())

    width = max(len(label) for label, _ in ROWS)
    header = f"| {'Metric'.ljust(width)} | " + " | ".join(a.ljust(16) for a in args.arms) + " |"
    print("\n" + header)
    print(f"|{'-' * (width + 2)}|" + "|".join("-" * 18 for _ in args.arms) + "|")
    for label, fmt in ROWS:
        cells = " | ".join(fmt(metrics[a]).ljust(16) for a in args.arms)
        print(f"| {label.ljust(width)} | {cells} |")

    (HERE / "results" / f"{args.tag}-comparison.json").write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
