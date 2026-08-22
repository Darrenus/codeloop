#!/usr/bin/env python3
"""Grade a predictions.json with the official SWE-bench harness and fold the
resolve rate back into that run's metrics.json.

Kept separate from run_swebench.py on purpose: generating patches and grading
them are different failure domains, and you re-grade far more often than you
re-run an agent.

    python eval/score.py --run-name sr-baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


def find_report(run_id: str, search_root: Path) -> Path | None:
    candidates = sorted(search_root.rglob(f"*{run_id}*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", choices=list(DATASETS), default="verified")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    run_dir = Path(__file__).parent / "results" / args.run_name
    preds = run_dir / "predictions.json"
    if not preds.exists():
        print(f"no predictions at {preds}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", DATASETS[args.split],
        "--predictions_path", str(preds),
        "--max_workers", str(args.workers),
        "--run_id", args.run_name,
    ]
    print(" ".join(cmd))
    if subprocess.run(cmd, cwd=run_dir).returncode != 0:
        print("evaluation harness failed", file=sys.stderr)
        return 1

    report = find_report(args.run_name, run_dir)
    if report is None:
        print("harness finished but no report json found; leaving metrics.json alone")
        return 0

    data = json.loads(report.read_text())
    resolved = data.get("resolved_instances", 0)
    total = data.get("total_instances", 0) or 1

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(
        {
            "resolved_instances": resolved,
            "total_instances": total,
            "resolve_rate": round(resolved / total, 4),
            "report_path": str(report.relative_to(run_dir)),
        }
    )
    metrics.pop("note", None)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"resolve rate: {resolved}/{total} = {resolved / total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
