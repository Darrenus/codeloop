"""A scripted stand-in for a model backend.

Lets the agent loop be tested end to end with no API key and no spend: you hand
it the turns you want the "model" to take and it replays them in order.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeloop.model import Reply  # noqa: E402

USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0, "cache_write_tokens": 0}


def text(body: str) -> dict:
    return {"type": "text", "text": body}


def tool(name: str, args: dict, id: str = "tu_1") -> dict:
    return {"type": "tool_use", "id": id, "name": name, "input": args}


def turn(*blocks: dict) -> Reply:
    """A turn ending in tool calls if any block is a tool use, else a final answer."""
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return Reply(list(blocks), stop, dict(USAGE))


class FakeModel:
    model = "fake-model"

    def __init__(self, script: List[Reply]):
        self._script = list(script)
        self.calls: List[dict] = []

    def complete(self, system: str, messages: List[dict], tools: List[dict]) -> Reply:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self._script:
            # Ran off the end of the script: end the conversation rather than
            # letting the loop spin.
            return turn(text("done"))
        return self._script.pop(0)
