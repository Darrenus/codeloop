"""A scripted stand-in for the Anthropic client.

Lets the agent loop be tested end to end with no API key and no spend: you hand
it the turns you want the "model" to take and it replays them in order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "tu_1"
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 100
    output_tokens: int = 20
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class Response:
    content: List[Any]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


def turn(*blocks) -> Response:
    """A turn ending in tool calls if any block is a tool use, else a final answer."""
    stop = "tool_use" if any(isinstance(b, ToolUseBlock) for b in blocks) else "end_turn"
    return Response(list(blocks), stop)


class FakeMessages:
    def __init__(self, script: List[Response]):
        self._script = list(script)
        self.calls: List[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            # Ran off the end of the script: end the conversation rather than
            # letting the loop spin.
            return turn(TextBlock("done"))
        return self._script.pop(0)


class FakeClient:
    def __init__(self, script: List[Response]):
        self.messages = FakeMessages(script)
