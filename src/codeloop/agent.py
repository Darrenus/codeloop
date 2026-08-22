"""The agent loop.

An agent is an LLM, a loop, and enough tokens. Everything this repo is actually
about lives *around* this file -- the environment seam, the edit format, the
approval policy, the metrics that make ablations comparable.

The message list is the trajectory. There is no second representation to keep
in sync, which is what makes a run trivially replayable.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import anthropic

from . import tools
from .env import Environment, LocalEnvironment

SYSTEM_PROMPT = """You are codeloop, a terminal coding agent working inside {cwd}.

Work in small, verifiable steps. Prefer `grep` and targeted `read_file` calls over
reading whole trees -- context you do not spend is context you still have. Before
editing a file, read the exact region you intend to change so your edit matches.
After editing, verify with `bash` (run the tests, or at minimum re-read the region).

When the task is complete, say so plainly and state how you verified it. Do not
ask the user questions; you are running unattended."""


class LimitExceeded(Exception):
    pass


class FatalAPIError(Exception):
    """An API failure no amount of retrying will fix -- bad key, no credit, a
    malformed request. Retrying these in a 30-instance batch just burns an hour
    to arrive at the same error."""


# 429 and 5xx are the ones worth waiting out; a long batch run will meet both.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class Usage:
    """Everything an ablation needs to compare two runs."""

    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    edit_attempts: int = 0
    edit_failures: int = 0
    tool_calls: dict = field(default_factory=dict)
    wall_seconds: float = 0.0

    @property
    def edit_failure_rate(self) -> float:
        return self.edit_failures / self.edit_attempts if self.edit_attempts else 0.0


class Agent:
    def __init__(
        self,
        env: Optional[Environment] = None,
        model: str = "claude-sonnet-5",
        edit_format: str = "search_replace",
        max_steps: int = 50,
        max_tokens: int = 8192,
        max_retries: int = 6,
        approve: Optional[Callable[[str, dict], bool]] = None,
        on_event: Optional[Callable[[str, object], None]] = None,
        client=None,
    ):
        # Injectable so the loop can be exercised against a scripted model with
        # no API key and no spend. The loop is the part most worth testing and
        # the part a live model makes hardest to test.
        self.client = client or anthropic.Anthropic()
        self.env = env or LocalEnvironment()
        self.model = model
        self.edit_format = edit_format
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.approve = approve or (lambda name, args: True)
        self.on_event = on_event or (lambda kind, payload: None)
        self.registry, self.schemas = tools.get_toolset(edit_format)
        self.messages: list[dict] = []
        self.usage = Usage()
        self.exit_reason = "unset"

    # -- main loop ---------------------------------------------------------
    def run(self, task: str) -> str:
        started = time.time()
        self.messages.append({"role": "user", "content": task})
        try:
            return self._loop()
        finally:
            self.usage.wall_seconds = round(time.time() - started, 1)

    def _loop(self) -> str:
        while True:
            if self.usage.steps >= self.max_steps:
                self.exit_reason = "step_limit"
                return "(step limit reached)"
            self.usage.steps += 1

            response = self._call_model()

            for block in response.content:
                if block.type == "text":
                    self.on_event("text", block.text)

            if response.stop_reason != "tool_use":
                self.exit_reason = "finished"
                return "".join(b.text for b in response.content if b.type == "text")

            results = [
                self._execute(b) for b in response.content if b.type == "tool_use"
            ]
            self.messages.append({"role": "user", "content": results})

    def _call_model(self):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._call_model_once()
            except FatalAPIError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status not in _RETRYABLE_STATUS:
                    raise FatalAPIError(f"{status}: {exc}") from exc
                if status is None and not _looks_transient(exc):
                    raise
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                # Full jitter: a batch run hits the rate limit on every worker at
                # once, and synchronised retries would just re-collide.
                delay = min(2 ** attempt, 60) * (0.5 + random.random() / 2)
                self.on_event("retry", (attempt + 1, round(delay, 1), str(exc)[:200]))
                time.sleep(delay)
        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_error}")

    def _call_model_once(self):
        # Cache the system prompt and tool definitions -- they are identical on
        # every one of the ~50 calls in a trajectory.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(cwd=self.env.cwd),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self.schemas,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        u = response.usage
        self.usage.input_tokens += u.input_tokens
        self.usage.output_tokens += u.output_tokens
        self.usage.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        return response

    # -- tool dispatch -----------------------------------------------------
    def _execute(self, block) -> dict:
        name, args = block.name, block.input
        self.on_event("tool_use", (name, args))
        self.usage.tool_calls[name] = self.usage.tool_calls.get(name, 0) + 1
        is_edit = name in ("edit_file", "write_file")
        if is_edit:
            self.usage.edit_attempts += 1

        def result(content: str, is_error: bool = False) -> dict:
            if is_error and is_edit:
                self.usage.edit_failures += 1
            self.on_event("tool_result", (name, content, is_error))
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content or "(empty)",
                "is_error": is_error,
            }

        if name not in self.registry:
            return result(f"unknown tool: {name}", is_error=True)
        if name not in tools.READ_ONLY and not self.approve(name, args):
            return result("The user declined this action.", is_error=True)
        try:
            return result(self.registry[name](self.env, **args))
        except Exception as exc:
            # Errors go back to the model rather than up the stack: a model that
            # can read its own failure usually fixes it on the next turn, and a
            # traceback ends the run.
            return result(f"{type(exc).__name__}: {exc}", is_error=True)

    # -- persistence -------------------------------------------------------
    def trajectory(self, **extra) -> dict:
        return {
            "model": self.model,
            "edit_format": self.edit_format,
            "exit_reason": self.exit_reason,
            "usage": asdict(self.usage),
            "messages": json.loads(json.dumps(self.messages, default=_encode)),
            **extra,
        }

    def dump_trajectory(self, path: str, **extra) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.trajectory(**extra), fh, indent=2)


def _looks_transient(exc: Exception) -> bool:
    """Connection resets and timeouts arrive without a status code."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(w in text for w in ("timeout", "connection", "temporarily", "overloaded"))


def _encode(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)
