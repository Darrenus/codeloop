from __future__ import annotations

import argparse
import sys

from .agent import Agent

DIM, BOLD, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m"


def _render(kind: str, payload) -> None:
    if kind == "text":
        print(payload)
    elif kind == "tool_use":
        name, args = payload
        preview = ", ".join(f"{k}={v!r}"[:80] for k, v in args.items())
        print(f"{DIM}→ {name}({preview}){RESET}")
    elif kind == "tool_result":
        name, content, is_error = payload
        colour = RED if is_error else DIM
        first = content.splitlines()[0] if content else ""
        print(f"{colour}  {first[:100]}{RESET}")


def _confirm(name: str, args: dict) -> bool:
    print(f"{YELLOW}Allow {BOLD}{name}{RESET}{YELLOW} with {args}? [y/N] {RESET}", end="")
    return input().strip().lower() in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(prog="codeloop")
    parser.add_argument("task", nargs="*", help="the task to work on")
    parser.add_argument("-m", "--model", default="claude-sonnet-5")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--yolo", action="store_true", help="skip approval for write actions")
    parser.add_argument("--trajectory", help="write the message history to this path")
    args = parser.parse_args()

    task = " ".join(args.task) or input("task> ")
    agent = Agent(
        model=args.model,
        max_steps=args.max_steps,
        approve=(lambda n, a: True) if args.yolo else _confirm,
        on_event=_render,
    )
    try:
        agent.run(task)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        if args.trajectory:
            agent.dump_trajectory(args.trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
