"""The agent loop.

An agent is an LLM, a loop, and enough tokens. Everything interesting in this
repo lives *around* this file — context management, approval policy, edit
formats — not inside it.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import anthropic

from . import tools

SYSTEM_PROMPT = """You are codeloop, a terminal coding agent working in the user's current directory.

Work in small, verifiable steps. Prefer `grep` and targeted `read_file` calls over
reading whole trees. Before editing a file, read the exact region you intend to
change so your `old_str` matches. After editing, run the project's tests or a
quick sanity check with `bash`.

When the task is done, state plainly what you changed and how you verified it."""


class Agent:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_steps: int = 50,
        approve: Optional[Callable[[str, dict], bool]] = None,
        on_event: Optional[Callable[[str, object], None]] = None,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_steps = max_steps
        self.approve = approve or (lambda name, args: True)
        self.on_event = on_event or (lambda kind, payload: None)
        self.messages: list[dict] = []

    def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})

        for _ in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=tools.SCHEMAS,
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            for block in response.content:
                if block.type == "text":
                    self.on_event("text", block.text)

            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self.on_event("tool_use", (block.name, block.input))
                results.append(self._execute(block))
            self.messages.append({"role": "user", "content": results})

        return "(step limit reached)"

    def _execute(self, block) -> dict:
        def result(content: str, is_error: bool = False) -> dict:
            self.on_event("tool_result", (block.name, content, is_error))
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            }

        if block.name not in tools.REGISTRY:
            return result(f"unknown tool: {block.name}", is_error=True)

        if block.name not in tools.READ_ONLY and not self.approve(block.name, block.input):
            return result("The user declined this action.", is_error=True)

        try:
            return result(tools.REGISTRY[block.name](**block.input))
        except Exception as exc:  # surfaced to the model so it can recover
            return result(f"{type(exc).__name__}: {exc}", is_error=True)

    def dump_trajectory(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.messages, fh, indent=2, default=str)
