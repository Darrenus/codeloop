"""Tests for the model seam: wire-format translation and completion caching.

The translation is the risky part of provider independence -- it is mechanical,
easy to get subtly wrong, and a mistake shows up as a model that mysteriously
never calls a tool.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeloop.model import PROVIDERS, CachedModel, OpenAICompatibleModel, Reply, build_model  # noqa: E402
from fake_model import FakeModel, text, tool, turn  # noqa: E402

TRANSLATE = OpenAICompatibleModel._to_openai_messages


class TranslationTests(unittest.TestCase):
    def test_system_prompt_leads(self):
        out = TRANSLATE("be helpful", [{"role": "user", "content": "hi"}])
        self.assertEqual({"role": "system", "content": "be helpful"}, out[0])
        self.assertEqual({"role": "user", "content": "hi"}, out[1])

    def test_assistant_tool_use_becomes_tool_calls(self):
        out = TRANSLATE("s", [
            {"role": "assistant", "content": [
                text("Let me look."),
                tool("read_file", {"path": "a.py"}, "t1"),
            ]},
        ])
        message = out[1]
        self.assertEqual("Let me look.", message["content"])
        call = message["tool_calls"][0]
        self.assertEqual("t1", call["id"])
        self.assertEqual("read_file", call["function"]["name"])
        # Arguments cross the wire as a JSON *string*, not an object.
        self.assertEqual({"path": "a.py"}, json.loads(call["function"]["arguments"]))

    def test_assistant_without_text_sends_null_content(self):
        out = TRANSLATE("s", [{"role": "assistant", "content": [tool("grep", {"pattern": "x"})]}])
        self.assertIsNone(out[1]["content"])

    def test_tool_results_become_one_tool_message_each(self):
        out = TRANSLATE("s", [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "bad", "is_error": True},
            ]},
        ])
        self.assertEqual(["system", "tool", "tool"], [m["role"] for m in out])
        self.assertEqual(["t1", "t2"], [m["tool_call_id"] for m in out[1:]])

    def test_tool_schemas_are_wrapped_as_functions(self):
        out = OpenAICompatibleModel._to_openai_tools([
            {"name": "grep", "description": "search", "input_schema": {"type": "object"}},
        ])
        self.assertEqual("function", out[0]["type"])
        self.assertEqual("grep", out[0]["function"]["name"])
        self.assertEqual({"type": "object"}, out[0]["function"]["parameters"])

    def test_round_trip_preserves_call_order(self):
        messages = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": [tool("read_file", {"path": "a"}, "t1"),
                                              tool("grep", {"pattern": "b"}, "t2")]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "A", "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "B", "is_error": False},
            ]},
        ]
        out = TRANSLATE("s", messages)
        self.assertEqual(["t1", "t2"], [c["id"] for c in out[2]["tool_calls"]])
        self.assertEqual(["t1", "t2"], [m["tool_call_id"] for m in out[3:]])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def cached(self, script):
        return CachedModel(FakeModel(script), self.tmp.name)

    def test_identical_request_is_served_from_disk(self):
        model = self.cached([turn(text("first")), turn(text("second"))])
        a = model.complete("s", [{"role": "user", "content": "hi"}], [])
        b = model.complete("s", [{"role": "user", "content": "hi"}], [])
        self.assertEqual("first", a.text)
        self.assertEqual("first", b.text)   # not "second" -- the backend was not called
        self.assertEqual(1, model.hits)
        self.assertEqual(1, model.misses)

    def test_replayed_reply_reports_no_usage(self):
        # Otherwise a free rerun would silently inflate the cost column.
        model = self.cached([turn(text("x"))])
        model.complete("s", [{"role": "user", "content": "hi"}], [])
        replayed = model.complete("s", [{"role": "user", "content": "hi"}], [])
        self.assertEqual({}, replayed.usage)

    def test_a_different_request_misses(self):
        model = self.cached([turn(text("first")), turn(text("second"))])
        model.complete("s", [{"role": "user", "content": "hi"}], [])
        other = model.complete("s", [{"role": "user", "content": "different"}], [])
        self.assertEqual("second", other.text)
        self.assertEqual(2, model.misses)

    def test_cache_survives_a_new_process(self):
        self.cached([turn(text("first"))]).complete("s", [{"role": "user", "content": "hi"}], [])
        fresh = self.cached([turn(text("would be second"))])
        self.assertEqual("first", fresh.complete("s", [{"role": "user", "content": "hi"}], []).text)
        self.assertEqual(1, fresh.hits)

    def test_tool_uses_round_trip_through_the_cache(self):
        model = self.cached([turn(tool("grep", {"pattern": "x"}, "t1"))])
        model.complete("s", [{"role": "user", "content": "hi"}], [])
        replayed = model.complete("s", [{"role": "user", "content": "hi"}], [])
        self.assertEqual("tool_use", replayed.stop_reason)
        self.assertEqual({"pattern": "x"}, replayed.tool_uses[0]["input"])


class BuildTests(unittest.TestCase):
    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model("not-a-provider")

    def test_missing_key_is_reported_before_any_request(self):
        with self.assertRaises(RuntimeError) as ctx:
            build_model("deepseek")
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_every_preset_is_complete(self):
        for name, preset in PROVIDERS.items():
            self.assertIn("base_url", preset, name)
            self.assertIn("default_model", preset, name)
            self.assertTrue(preset["default_model"], name)

    def test_local_provider_needs_no_key(self):
        self.assertIsNone(PROVIDERS["ollama"]["key_env"])


class ReplyTests(unittest.TestCase):
    def test_text_concatenates_and_tool_uses_filter(self):
        reply = Reply([text("a"), tool("grep", {}, "t"), text("b")], "tool_use")
        self.assertEqual("ab", reply.text)
        self.assertEqual(1, len(reply.tool_uses))


if __name__ == "__main__":
    unittest.main()


class ReplayTests(unittest.TestCase):
    """Replay is the only mechanism that makes harness iteration reliably free,
    since the providers are not deterministic enough for a request-hash cache to
    survive a whole trajectory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "traj.json"
        self.path.write_text(json.dumps({"messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": [tool("read_file", {"path": "a.py"}, "t1")]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "…", "is_error": False}]},
            {"role": "assistant", "content": [text("done")]},
        ]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_replays_assistant_turns_in_order(self):
        from codeloop.model import ReplayModel

        model = ReplayModel(str(self.path))
        first = model.complete("s", [], [])
        self.assertEqual("tool_use", first.stop_reason)
        self.assertEqual("read_file", first.tool_uses[0]["name"])

        second = model.complete("s", [], [])
        self.assertEqual("end_turn", second.stop_reason)
        self.assertEqual("done", second.text)

    def test_request_is_ignored(self):
        from codeloop.model import ReplayModel

        model = ReplayModel(str(self.path))
        # A completely different request still gets the recording's first turn.
        reply = model.complete("other system", [{"role": "user", "content": "unrelated"}], [])
        self.assertEqual("read_file", reply.tool_uses[0]["name"])

    def test_replayed_turns_report_no_usage(self):
        from codeloop.model import ReplayModel

        self.assertEqual({}, ReplayModel(str(self.path)).complete("s", [], []).usage)

    def test_running_past_the_recording_is_an_explicit_error(self):
        from codeloop.model import ReplayExhausted, ReplayModel

        model = ReplayModel(str(self.path))
        model.complete("s", [], [])
        model.complete("s", [], [])
        with self.assertRaises(ReplayExhausted):
            model.complete("s", [], [])
