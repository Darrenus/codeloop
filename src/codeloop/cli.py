from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent, FatalAPIError
from .env import DockerEnvironment, LocalEnvironment
from .model import PROVIDERS, ReplayModel, build_model
from .tools import EDIT_FORMATS

DIM, BOLD, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[31m", "\033[0m"


def load_dotenv(start: Path) -> None:
    """Read KEY=VALUE lines from the nearest .env, walking upwards.

    Keys for a dozen providers do not belong in a shell profile, and an agent
    that edits files should not be the reason a key ends up committed.
    """
    for directory in [start, *start.parents]:
        path = directory / ".env"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


def _render(kind: str, payload) -> None:
    if kind == "text":
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
    p.add_argument("-p", "--provider", default=os.environ.get("CODELOOP_PROVIDER", "anthropic"),
                   choices=sorted(PROVIDERS), help="model provider")
    p.add_argument("-m", "--model", default=os.environ.get("CODELOOP_MODEL"),
                   help="model name (defaults to the provider's)")
    p.add_argument("--edit-format", choices=EDIT_FORMATS, default="search_replace")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--cache-dir", help="record and replay completions from this directory")
    p.add_argument("--replay", metavar="TRAJECTORY",
                   help="replay a recorded trajectory's turns instead of calling a model (free)")
    p.add_argument("--docker-image", help="run inside this container instead of locally")
    p.add_argument("--yolo", action="store_true", help="skip approval for write actions")
    p.add_argument("--trajectory", help="write the trajectory JSON to this path")
    p.add_argument("--list-providers", action="store_true")
    args = p.parse_args()

    if args.list_providers:
        for name, preset in sorted(PROVIDERS.items()):
            key = preset["key_env"] or "-"
            mark = "set" if not preset["key_env"] or os.environ.get(preset["key_env"]) else "unset"
            print(f"{name:<12} {preset['default_model']:<40} {key} ({mark})")
        return 0

    load_dotenv(Path.cwd())
    task = " ".join(args.task) or input("task> ")

    try:
        model = (
            ReplayModel(args.replay) if args.replay
            else build_model(args.provider, args.model, args.cache_dir)
        )
    except (ValueError, RuntimeError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        print("Run `codeloop --list-providers` to see what is configured.", file=sys.stderr)
        return 2

    env = DockerEnvironment(args.docker_image) if args.docker_image else LocalEnvironment()
    agent = Agent(
        model=model,
        env=env,
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
        # A raw traceback here tells the user nothing they can act on.
        print(f"\n{RED}API error: {exc}{RESET}", file=sys.stderr)
        text = str(exc).lower()
        if "credit" in text or "balance" in text or "quota" in text:
            print(
                "Out of credit for this provider. `codeloop --list-providers` shows the\n"
                "alternatives; several have free tiers, and `-p ollama` needs no key at all.",
                file=sys.stderr,
            )
        elif "authentication" in text or "401" in text or "api key" in text:
            print(f"Check the API key for provider {args.provider!r}.", file=sys.stderr)
        exit_code = 2
    finally:
        u = agent.usage
        cache_note = ""
        if hasattr(model, "hits"):
            cache_note = f" · cache {model.hits} hit / {model.misses} miss"
        print(
            f"{DIM}--- {u.steps} steps · {u.input_tokens:,} in / {u.output_tokens:,} out"
            f" · edits {u.edit_attempts - u.edit_failures}/{u.edit_attempts} ok"
            f" · {u.wall_seconds}s{cache_note}{RESET}"
        )
        if args.trajectory:
            agent.dump_trajectory(args.trajectory, task=task)
        env.cleanup()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
