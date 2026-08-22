"""Model backends.

The agent speaks one canonical message format -- Anthropic's block shape, which
is the most expressive of the common ones -- and each backend translates to and
from its provider's wire format. Hardcoding a single vendor into the loop would
tie the project's running cost to one price list and make the loop untestable
without a paid account; both turned out to matter.

Everything here returns a `Reply`, which is plain JSON-serialisable data. That
is what lets `CachedModel` record and replay a run for free.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Reply:
    blocks: List[dict]
    """Canonical blocks: {"type": "text"|"tool_use", ...}."""
    stop_reason: str
    """Either "tool_use" or "end_turn"."""
    usage: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.blocks if b["type"] == "text")

    @property
    def tool_uses(self) -> List[dict]:
        return [b for b in self.blocks if b["type"] == "tool_use"]


# ---------------------------------------------------------------------------
# Provider presets
#
# Every entry below except "anthropic" speaks the OpenAI chat-completions
# protocol, which by now is the lingua franca -- including for local servers.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "anthropic":   {"base_url": None,                                                    "key_env": "ANTHROPIC_API_KEY",  "default_model": "claude-sonnet-5"},
    "deepseek":    {"base_url": "https://api.deepseek.com/v1",                           "key_env": "DEEPSEEK_API_KEY",   "default_model": "deepseek-chat"},
    "qwen":        {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",     "key_env": "DASHSCOPE_API_KEY",  "default_model": "qwen-plus"},
    "glm":         {"base_url": "https://open.bigmodel.cn/api/paas/v4",                  "key_env": "ZHIPU_API_KEY",      "default_model": "glm-4-flash"},
    "moonshot":    {"base_url": "https://api.moonshot.cn/v1",                            "key_env": "MOONSHOT_API_KEY",   "default_model": "moonshot-v1-8k"},
    "siliconflow": {"base_url": "https://api.siliconflow.cn/v1",                         "key_env": "SILICONFLOW_API_KEY","default_model": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    "groq":        {"base_url": "https://api.groq.com/openai/v1",                        "key_env": "GROQ_API_KEY",       "default_model": "llama-3.3-70b-versatile"},
    "openrouter":  {"base_url": "https://openrouter.ai/api/v1",                          "key_env": "OPENROUTER_API_KEY", "default_model": "deepseek/deepseek-chat-v3:free"},
    "gemini":      {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",  "default_model": "gemini-flash-latest"},
    "cerebras":    {"base_url": "https://api.cerebras.ai/v1",                           "key_env": "CEREBRAS_API_KEY",   "default_model": "llama-3.3-70b"},
    "ollama":      {"base_url": "http://localhost:11434/v1",                             "key_env": None,                 "default_model": "qwen2.5-coder:7b"},
}


def build_model(provider: str, model: Optional[str] = None, cache_dir: Optional[str] = None):
    """Construct a backend from a provider preset, optionally wrapped in a cache."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}")
    preset = PROVIDERS[provider]
    name = model or preset["default_model"]
    key = os.environ.get(preset["key_env"], "") if preset["key_env"] else "not-needed"
    if not key and preset["key_env"]:
        raise RuntimeError(f"{preset['key_env']} is not set (provider {provider!r})")

    if provider == "anthropic":
        backend = AnthropicModel(name, api_key=key)
    else:
        backend = OpenAICompatibleModel(name, base_url=preset["base_url"], api_key=key)

    return CachedModel(backend, cache_dir) if cache_dir else backend


class AnthropicModel:
    def __init__(self, model: str, api_key: Optional[str] = None, max_tokens: int = 8192):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.supports_cache_control = True

    def complete(self, system: str, messages: List[dict], tools: List[dict]) -> Reply:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt and tool schemas are byte-identical across every
            # call in a trajectory, so they are worth caching.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        blocks = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                blocks.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
        u = response.usage
        return Reply(
            blocks=blocks,
            stop_reason="tool_use" if response.stop_reason == "tool_use" else "end_turn",
            usage={
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            },
        )


class OpenAICompatibleModel:
    """Any endpoint speaking OpenAI chat-completions: DeepSeek, Qwen, GLM,
    Moonshot, Groq, OpenRouter, Gemini's compatibility endpoint, GitHub Models,
    and local Ollama or vLLM servers."""

    def __init__(self, model: str, base_url: str, api_key: str, max_tokens: int = 8192):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.model = model
        self.max_tokens = max_tokens
        self.supports_cache_control = False

    # -- format translation ------------------------------------------------
    @staticmethod
    def _to_openai_messages(system: str, messages: List[dict]) -> List[dict]:
        out = [{"role": "system", "content": system}]
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue

            if msg["role"] == "assistant":
                text = "".join(b.get("text", "") for b in content if b["type"] == "text")
                calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in content if b["type"] == "tool_use"
                ]
                entry = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
            else:
                # A user turn carrying tool results becomes one "tool" message
                # per result, which is how the OpenAI protocol models it.
                for b in content:
                    if b.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": str(b["content"]),
                        })
                    elif b.get("type") == "text":
                        out.append({"role": "user", "content": b["text"]})
        return out

    @staticmethod
    def _to_openai_tools(tools: List[dict]) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def complete(self, system: str, messages: List[dict], tools: List[dict]) -> Reply:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._to_openai_messages(system, messages),
            tools=self._to_openai_tools(tools),
        )
        choice = response.choices[0]
        blocks: List[dict] = []
        if choice.message.content:
            blocks.append({"type": "text", "text": choice.message.content})
        for call in choice.message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # Weaker models emit malformed JSON often enough that it has to
                # be a tool error the model can see and retry, not a crash.
                blocks.append({
                    "type": "tool_use", "id": call.id, "name": call.function.name,
                    "input": {"__malformed__": call.function.arguments},
                })
                continue
            blocks.append({
                "type": "tool_use", "id": call.id,
                "name": call.function.name, "input": arguments,
            })

        u = response.usage
        cached = 0
        if u is not None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
        return Reply(
            blocks=blocks,
            stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
            usage={
                "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
                "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
                "cache_read_tokens": cached,
                "cache_write_tokens": 0,
            },
        )


class CachedModel:
    """Records every reply to disk and replays it on an identical request.

    Iterating on the harness -- patch extraction, metrics, the ablation table --
    otherwise means paying for the same completions again on every run. With a
    cache the first pass costs money and every pass after it is free, which also
    makes a published result exactly reproducible.
    """

    def __init__(self, backend, cache_dir: str):
        self.backend = backend
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self.backend.model

    def _key(self, system: str, messages: List[dict], tools: List[dict]) -> str:
        payload = json.dumps(
            {"m": self.backend.model, "s": system, "msgs": messages, "t": tools},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def complete(self, system: str, messages: List[dict], tools: List[dict]) -> Reply:
        path = self.dir / f"{self._key(system, messages, tools)}.json"
        if path.exists():
            self.hits += 1
            data = json.loads(path.read_text())
            # Replay is free, so a replayed reply reports zero usage; otherwise
            # a cached rerun would silently inflate the cost column.
            return Reply(data["blocks"], data["stop_reason"], {})
        self.misses += 1
        reply = self.backend.complete(system, messages, tools)
        path.write_text(json.dumps(
            {"blocks": reply.blocks, "stop_reason": reply.stop_reason, "usage": reply.usage},
            indent=2,
        ))
        return reply
