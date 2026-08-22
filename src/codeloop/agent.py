"""The agent loop.

An agent is an LLM, a loop, and enough tokens. Everything this repo is actually
about lives *around* this file -- the model seam, the environment seam, the edit
format, the approval policy, the metrics that make ablations comparable.

The message list is the trajectory, in one canonical provider-neutral format.
There is no second representation to keep in sync, which is what makes a run
replayable and gradeable.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from . import tools
from .env import Environment, LocalEnvironment

SYSTEM_PROMPT = """You are codeloop, a terminal coding agent working inside {cwd}.

Work in small, verifiable steps. Prefer `grep` and targeted `read_file` calls over
reading whole trees -- context you do not spend is context you still have. Before
editing a file, read the exact region you intend to change so your edit matches.
After editing, verify with `bash` (run the tests, or at minimum re-read the region).

Call tools using the tool-calling interface, never by describing the call in prose.
When the task is complete, say so plainly and state how you verified it. Do not ask
the user questions; you are running unattended."""


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

    def add(self, other: dict) -> None:
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            setattr(self, key, getattr(self, key) + other.get(key, 0))


class Agent:
    def __init__(
        self,
        model,
        env: Optional[Environment] = None,
        edit_format: str = "search_replace",
        max_steps: int = 50,
        max_retries: int = 6,
        approve: Optional[Callable[[str, dict], bool]] = None,
        on_event: Optional[Callable[[str, object], None]] = None,
    ):
        self.model = model
        self.env = env or LocalEnvironment()
        self.edit_format = edit_format
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.approve = approve or (lambda name, args: True)
        self.on_event = on_event or (lambda kind, payload: None)
        self.registry, self.schemas = tools.get_toolset(edit_format)
        self.messages: List[dict] = []
        self.usage = Usage()
        self.exit_reason = "unset"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(cwd=self.env.cwd)

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

            reply = self._complete()
            self.messages.append({"role": "assistant", "content": reply.blocks})
            if reply.text.strip():
                self.on_event("text", reply.text)

            if reply.stop_reason != "tool_use":
                self.exit_reason = "finished"
                return reply.text

            results = [self._execute(b) for b in reply.tool_uses]
            self.messages.append({"role": "user", "content": results})

    def _complete(self):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                reply = self.model.complete(self.system_prompt, self.messages, self.schemas)
                self.usage.add(reply.usage)
                return reply
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
                # once, and synchronised retries would just re-collide. Free tiers
                # rate-limit hard, so this path is well travelled.
                delay = min(2 ** attempt, 60) * (0.5 + random.random() / 2)
                self.on_event("retry", (attempt + 1, round(delay, 1), str(exc)[:200]))
                time.sleep(delay)
        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_error}")

    # -- tool dispatch -----------------------------------------------------
    def _execute(self, block: dict) -> dict:
        name, args = block["name"], block.get("input") or {}
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
                "tool_use_id": block["id"],
                "content": content or "(empty)",
                "is_error": is_error,
            }

        if "__malformed__" in args:
            # Weaker models emit invalid JSON arguments; that is a fact about the
            # model worth measuring, not a reason to abort the trajectory.
            return result("Tool arguments were not valid JSON. Re-issue the call.", is_error=True)
        if name not in self.registry:
            return result(f"unknown tool: {name}", is_error=True)
        if name not in tools.READ_ONLY and not self.approve(name, args):
            return result("The user declined this action.", is_error=True)
        try:
            return result(self.registry[name](self.env, **args))
        except TypeError as exc:
            return result(f"Bad arguments for {name}: {exc}", is_error=True)
        except Exception as exc:
            # Errors go back to the model rather than up the stack: a model that
            # can read its own failure usually fixes it on the next turn, and a
            # traceback ends the run.
            return result(f"{type(exc).__name__}: {exc}", is_error=True)

    # -- persistence -------------------------------------------------------
    def trajectory(self, **extra) -> dict:
        return {
            "model": getattr(self.model, "model", str(self.model)),
            "edit_format": self.edit_format,
            "exit_reason": self.exit_reason,
            "usage": asdict(self.usage),
            "messages": self.messages,
            **extra,
        }

    def dump_trajectory(self, path: str, **extra) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.trajectory(**extra), fh, indent=2, default=str)


def _looks_transient(exc: Exception) -> bool:
    """Connection resets and timeouts arrive without a status code."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(w in text for w in ("timeout", "connection", "temporarily", "overloaded"))
