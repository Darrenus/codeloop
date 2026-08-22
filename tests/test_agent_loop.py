"""End-to-end tests of the agent loop against a scripted model.

No API key, no spend, no Docker. These cover the wiring that a live model makes
awkward to test: usage accounting, the approval gate, edit-failure attribution,
the step limit, and error recovery.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codeloop.agent import Agent, FatalAPIError  # noqa: E402
from codeloop.env import LocalEnvironment  # noqa: E402
from fake_model import FakeModel, text, tool, turn  # noqa: E402


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = LocalEnvironment(cwd=self.tmp.name)
        Path(self.tmp.name, "calc.py").write_text("def add(a, b):\n    return a - b\n")

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, script, **kwargs):
        return Agent(model=FakeModel(script), env=self.env, **kwargs)

    def test_full_read_edit_verify_cycle(self):
        agent = self.agent([
            turn(tool("read_file", {"path": "calc.py"})),
            turn(tool("edit_file", {
                "path": "calc.py", "old_str": "return a - b", "new_str": "return a + b",
            })),
            turn(tool("bash", {"command": "python3 -c 'import calc; print(calc.add(2,3))'"})),
            turn(text("Fixed the sign error; add(2,3) now returns 5.")),
        ])
        answer = agent.run("add() subtracts instead of adding")

        self.assertIn("Fixed the sign error", answer)
        self.assertEqual("finished", agent.exit_reason)
        self.assertIn("return a + b", Path(self.tmp.name, "calc.py").read_text())
        self.assertEqual(4, agent.usage.steps)
        self.assertEqual(1, agent.usage.edit_attempts)
        self.assertEqual(0, agent.usage.edit_failures)
        self.assertEqual(
            {"read_file": 1, "edit_file": 1, "bash": 1}, agent.usage.tool_calls
        )

    def test_usage_accumulates_across_turns(self):
        agent = self.agent([
            turn(tool("list_files", {"path": "."})),
            turn(text("done")),
        ])
        agent.run("look around")
        self.assertEqual(200, agent.usage.input_tokens)   # 2 calls x 100
        self.assertEqual(40, agent.usage.output_tokens)   # 2 calls x 20

    def test_failed_edit_is_counted_and_reported_to_the_model(self):
        agent = self.agent([
            turn(tool("edit_file", {
                "path": "calc.py", "old_str": "nonexistent", "new_str": "x",
            })),
            turn(text("giving up")),
        ])
        agent.run("break something")

        self.assertEqual(1, agent.usage.edit_attempts)
        self.assertEqual(1, agent.usage.edit_failures)
        self.assertEqual(1.0, agent.usage.edit_failure_rate)
        # The failure must reach the model as a tool_result, not raise.
        results = agent.messages[-2]["content"]
        self.assertTrue(results[0]["is_error"])
        self.assertIn("old_str not found", results[0]["content"])

    def test_approval_denial_reaches_the_model_and_blocks_the_write(self):
        agent = self.agent(
            [
                turn(tool("edit_file", {
                    "path": "calc.py", "old_str": "return a - b", "new_str": "return 0",
                })),
                turn(text("understood")),
            ],
            approve=lambda name, args: False,
        )
        agent.run("wreck it")

        self.assertIn("return a - b", Path(self.tmp.name, "calc.py").read_text())
        self.assertIn("declined", agent.messages[-2]["content"][0]["content"])

    def test_read_only_tools_bypass_approval(self):
        seen = []
        agent = self.agent(
            [turn(tool("grep", {"pattern": "def"})), turn(text("ok"))],
            approve=lambda name, args: seen.append(name) or True,
        )
        agent.run("find the functions")
        self.assertEqual([], seen)

    def test_step_limit_stops_the_loop(self):
        script = [turn(tool("list_files", {"path": "."})) for _ in range(10)]
        agent = self.agent(script, max_steps=3)
        agent.run("loop forever")
        self.assertEqual("step_limit", agent.exit_reason)
        self.assertEqual(3, agent.usage.steps)

    def test_unknown_tool_is_reported_not_raised(self):
        agent = self.agent([
            turn(tool("teleport", {"x": 1})),
            turn(text("ok")),
        ])
        agent.run("do the impossible")
        self.assertIn("unknown tool", agent.messages[-2]["content"][0]["content"])

    def test_parallel_tool_calls_in_one_turn(self):
        agent = self.agent([
            turn(
                tool("read_file", {"path": "calc.py"}, "a"),
                tool("list_files", {"path": "."}, "b"),
            ),
            turn(text("ok")),
        ])
        agent.run("look at both")
        results = agent.messages[-2]["content"]
        self.assertEqual(["a", "b"], [r["tool_use_id"] for r in results])

    def test_whole_file_arm_exposes_write_file_only(self):
        agent = self.agent([turn(text("ok"))], edit_format="whole_file")
        agent.run("noop")
        names = {s["name"] for s in agent.schemas}
        self.assertIn("write_file", names)
        self.assertNotIn("edit_file", names)

    def test_system_prompt_carries_the_workspace_root(self):
        agent = self.agent([turn(text("ok"))])
        agent.run("noop")
        self.assertIn(self.tmp.name, agent.model.calls[0]["system"])

    def test_trajectory_round_trips_to_json(self):
        agent = self.agent([
            turn(tool("read_file", {"path": "calc.py"})),
            turn(text("ok")),
        ])
        agent.run("read it")
        out = Path(self.tmp.name, "traj.json")
        agent.dump_trajectory(str(out), instance_id="demo")

        data = json.loads(out.read_text())
        self.assertEqual("demo", data["instance_id"])
        self.assertEqual("search_replace", data["edit_format"])
        self.assertEqual("finished", data["exit_reason"])
        self.assertEqual(2, data["usage"]["steps"])
        self.assertTrue(data["messages"])


if __name__ == "__main__":
    unittest.main()


class _Boom(Exception):
    def __init__(self, status=None, msg="boom"):
        super().__init__(msg)
        self.status_code = status


class RetryTests(unittest.TestCase):
    """A 30-instance batch will meet 429s and connection resets; it must not
    meet them by dying, and it must not wait out a failure that is permanent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = LocalEnvironment(cwd=self.tmp.name)
        self.slept = []

    def tearDown(self):
        self.tmp.cleanup()

    def agent_with(self, side_effects, **kwargs):
        from codeloop import agent as agent_mod

        model = FakeModel([turn(text("ok"))])
        calls = {"n": 0}
        real_complete = model.complete

        def complete(*a, **kw):
            i = calls["n"]
            calls["n"] += 1
            if i < len(side_effects) and side_effects[i] is not None:
                raise side_effects[i]
            return real_complete(*a, **kw)

        model.complete = complete
        agent_mod.time.sleep = lambda d: self.slept.append(d)
        return Agent(model=model, env=self.env, **kwargs), calls

    def test_retries_then_succeeds(self):
        agent, calls = self.agent_with([_Boom(429), _Boom(503), None])
        agent.run("go")
        self.assertEqual(3, calls["n"])
        self.assertEqual(2, len(self.slept))

    def test_backoff_grows_and_is_jittered(self):
        agent, _ = self.agent_with([_Boom(429), _Boom(429), _Boom(429), None])
        agent.run("go")
        self.assertLess(self.slept[0], self.slept[-1])
        self.assertTrue(all(d > 0 for d in self.slept))

    def test_permanent_error_is_not_retried(self):
        agent, calls = self.agent_with([_Boom(400, "credit balance is too low")])
        with self.assertRaises(FatalAPIError) as ctx:
            agent.run("go")
        self.assertEqual(1, calls["n"])
        self.assertEqual([], self.slept)
        self.assertIn("credit balance", str(ctx.exception))

    def test_auth_error_is_not_retried(self):
        agent, calls = self.agent_with([_Boom(401, "authentication_error")])
        with self.assertRaises(FatalAPIError):
            agent.run("go")
        self.assertEqual(1, calls["n"])

    def test_connection_error_without_status_is_retried(self):
        agent, calls = self.agent_with([_Boom(None, "Connection reset by peer"), None])
        agent.run("go")
        self.assertEqual(2, calls["n"])

    def test_gives_up_after_max_retries(self):
        agent, calls = self.agent_with([_Boom(429)] * 10, max_retries=3)
        with self.assertRaises(RuntimeError):
            agent.run("go")
        self.assertEqual(3, calls["n"])
        self.assertEqual(2, len(self.slept))
