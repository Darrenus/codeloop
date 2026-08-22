from __future__ import annotations

import argparse
import sys

from .agent import Agent, FatalAPIError
from .env import DockerEnvironment, LocalEnvironment
from .tools import EDIT_FORMATS

DIM, BOLD, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m"


def _render(kind: str, payload) -> None:
    if kind == "text":
        if payload.strip():
            print(payload)
    elif kind == "tool_use":
        name, args = payload
        preview = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())
        print(f"{DIM}→ {name}({preview}){RESET}")
    elif kind == "retry":
        attempt, delay, message = payload
        print(f"{YELLOW}  retry {attempt} in {delay}s — {message}{RESET}", file=sys.stderr)
    elif kind == "tool_result":
        name, content, is_error = payload
        first = content.splitlines()[0] if content else ""
        print(f"{RED if is_error else DIM}  {first[:110]}{RESET}")


def _confirm(name: str, args: dict) -> bool:
    print(f"{YELLOW}Allow {BOLD}{name}{RESET}{YELLOW} {args}? [y/N] {RESET}", end="")
    return input().strip().lower() in {"y", "yes"}


def main() -> int:
    p = argparse.ArgumentParser(prog="codeloop")
    p.add_argument("task", nargs="*", help="the task to work on")
    p.add_argument("-m", "--model", default="claude-sonnet-5")
    p.add_argument("--edit-format", choices=EDIT_FORMATS, default="search_replace")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--docker-image", help="run inside this container instead of locally")
    p.add_argument("--yolo", action="store_true", help="skip approval for write actions")
    p.add_argument("--trajectory", help="write the trajectory JSON to this path")
    args = p.parse_args()

    task = " ".join(args.task) or input("task> ")
    env = DockerEnvironment(args.docker_image) if args.docker_image else LocalEnvironment()
    agent = Agent(
        env=env,
        model=args.model,
        edit_format=args.edit_format,
        max_steps=args.max_steps,
        approve=(lambda n, a: True) if args.yolo else _confirm,
        on_event=_render,
    )
    exit_code = 0
    try:
        agent.run(task)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        exit_code = 130
    except FatalAPIError as exc:
        # A raw traceback here tells the user nothing they can act on; the two
        # failures they will actually hit are a bad key and an empty balance.
        print(f"\n{RED}API error: {exc}{RESET}", file=sys.stderr)
        text = str(exc)
        if "credit balance" in text:
            print(
                "Add credits at console.anthropic.com under Plans & Billing. "
                "Note that a Claude Pro/Max subscription is billed separately "
                "and does not fund API usage.",
                file=sys.stderr,
            )
        elif "authentication" in text.lower() or "401" in text:
            print("Check ANTHROPIC_API_KEY.", file=sys.stderr)
        exit_code = 2
    finally:
        u = agent.usage
        print(
            f"{DIM}--- {u.steps} steps · {u.input_tokens:,} in / {u.output_tokens:,} out"
            f" · {u.cache_read_tokens:,} cached · edits {u.edit_attempts - u.edit_failures}"
            f"/{u.edit_attempts} ok · {u.wall_seconds}s{RESET}"
        )
        if args.trajectory:
            agent.dump_trajectory(args.trajectory, task=task)
        env.cleanup()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
