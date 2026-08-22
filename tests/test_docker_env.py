"""Integration tests for DockerEnvironment.

Costs nothing: no model is called. These exercise the seam the SWE-bench harness
depends on -- a container that keeps filesystem state across `docker exec` calls,
the tool surface running inside it, and the `git diff` patch extraction.

Skipped automatically when Docker is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeloop import tools  # noqa: E402
from codeloop.env import DockerEnvironment  # noqa: E402
from codeloop.patch import extract_patch  # noqa: E402

IMAGE = os.environ.get("CODELOOP_TEST_IMAGE", "python:3.11")
# Native by default so the suite is quick; set to linux/amd64 to exercise the
# emulation path that the real SWE-bench images require.
PLATFORM = os.environ.get("CODELOOP_TEST_PLATFORM", "")


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def default_platform() -> str:
    if PLATFORM:
        return PLATFORM
    arch = subprocess.run(
        ["docker", "info", "--format", "{{.Architecture}}"],
        capture_output=True, text=True,
    ).stdout.strip()
    return "linux/arm64" if arch in ("aarch64", "arm64") else "linux/amd64"


@unittest.skipUnless(docker_available(), "docker daemon not available")
class DockerEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DockerEnvironment(IMAGE, cwd="/work", platform=default_platform())
        cls.env.execute("mkdir -p /work")

    @classmethod
    def tearDownClass(cls):
        cls.env.cleanup()

    def test_container_is_linux(self):
        self.assertIn("Linux", self.env.execute("uname -s")["output"])

    def test_state_persists_across_exec_calls(self):
        # The whole reason the container is long-lived rather than one-shot.
        self.env.execute("echo persisted > /work/marker.txt")
        self.assertIn("persisted", self.env.execute("cat /work/marker.txt")["output"])

    def test_cwd_is_honoured(self):
        self.assertEqual("/work", self.env.execute("pwd")["output"].strip())

    def test_returncode_propagates(self):
        self.assertEqual(7, self.env.execute("exit 7")["returncode"])

    def test_timeout_is_reported(self):
        result = self.env.execute("sleep 30", timeout=2)
        self.assertEqual(124, result["returncode"])

    def test_output_is_clipped(self):
        result = self.env.execute("python3 -c \"print('x' * 200000)\"")
        self.assertIn("characters omitted", result["output"])

    def test_tools_run_inside_container(self):
        self.env.execute("printf 'def f():\\n    return 1\\n' > /work/a.py")
        self.assertIn("1\tdef f():", tools.read_file(self.env, "a.py"))
        self.assertIn("a.py", tools.list_files(self.env, "."))
        self.assertIn("return 1", tools.grep(self.env, "return", "."))

    def test_edit_file_inside_container(self):
        self.env.execute("printf 'x = 1\\n' > /work/b.py")
        tools.edit_file(self.env, "b.py", "x = 1", "x = 2")
        self.assertIn("x = 2", self.env.execute("cat /work/b.py")["output"])

    def test_edit_survives_shell_metacharacters_inside_container(self):
        # base64 transport is what makes this safe; a naive heredoc would mangle it.
        self.env.execute("""printf 'echo $(id) `pwd` "q"\\n' > /work/c.sh""")
        tools.edit_file(self.env, "c.sh", "$(id)", "'literal $VAR'")
        self.assertIn("'literal $VAR'", self.env.execute("cat /work/c.sh")["output"])

    def test_write_file_inside_container(self):
        tools.write_file(self.env, "d/e.py", "# whole file\n")
        self.assertIn("# whole file", self.env.execute("cat /work/d/e.py")["output"])

    def test_ambiguous_edit_is_refused_inside_container(self):
        self.env.execute("printf 'y = 1\\ny = 1\\n' > /work/f.py")
        with self.assertRaises(RuntimeError) as ctx:
            tools.edit_file(self.env, "f.py", "y = 1", "y = 2")
        self.assertIn("appears 2 times", str(ctx.exception))


@unittest.skipUnless(docker_available(), "docker daemon not available")
class PatchExtractionTests(unittest.TestCase):
    """Rehearses exactly what eval/run_swebench.py does to produce a submission."""

    def setUp(self):
        self.env = DockerEnvironment(IMAGE, cwd="/testbed", platform=default_platform())
        self.env.execute(
            "mkdir -p /testbed && cd /testbed && git init -q "
            "&& git config user.email t@t && git config user.name t "
            "&& printf 'def add(a, b):\\n    return a - b\\n' > calc.py "
            "&& git add -A && git commit -q -m base"
        )
        self.env.execute("git config --global --add safe.directory /testbed")

    def tearDown(self):
        self.env.cleanup()

    def test_edit_then_diff_yields_a_patch(self):
        tools.edit_file(self.env, "calc.py", "return a - b", "return a + b")
        patch = extract_patch(self.env)
        self.assertIn("--- a/calc.py", patch)
        self.assertIn("-    return a - b", patch)
        self.assertIn("+    return a + b", patch)

    def test_new_file_appears_in_patch(self):
        tools.write_file(self.env, "helper.py", "VALUE = 1\n")
        patch = extract_patch(self.env)
        self.assertIn("helper.py", patch)
        self.assertIn("new file mode", patch)

    def test_no_edit_yields_empty_patch(self):
        self.assertEqual("", extract_patch(self.env).strip())

    def test_build_artefacts_are_kept_out_of_the_patch(self):
        # Verifying a fix by importing the module is the normal thing for an
        # agent to do, and it litters the tree with bytecode. None of it belongs
        # in a submission.
        tools.edit_file(self.env, "calc.py", "return a - b", "return a + b")
        self.env.execute("python3 -c 'import calc'")
        self.assertIn("__pycache__", self.env.execute("ls -a")["output"])

        patch = extract_patch(self.env)
        self.assertNotIn("__pycache__", patch)
        self.assertNotIn(".pyc", patch)
        self.assertIn("+    return a + b", patch)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(docker_available(), "docker daemon not available")
class AgentInContainerTests(unittest.TestCase):
    """The full run_swebench.run_instance path, with the model scripted.

    Everything the real harness does except calling Claude: a container built
    from an image, an agent editing inside it, and a submission patch read back
    out of git.
    """

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        self.env = DockerEnvironment(IMAGE, cwd="/testbed", platform=default_platform())
        self.env.execute(
            "cd /testbed && git init -q && git config user.email t@t && git config user.name t "
            "&& printf 'def add(a, b):\\n    return a - b\\n' > calc.py "
            "&& git add -A && git commit -q -m base"
        )
        self.env.execute("git config --global --add safe.directory /testbed")

    def tearDown(self):
        self.env.cleanup()

    def test_agent_produces_an_applicable_patch(self):
        from codeloop.agent import Agent
        from fake_model import FakeModel, text, tool, turn

        agent = Agent(
            env=self.env,
            model=FakeModel([
                turn(tool("read_file", {"path": "calc.py"})),
                turn(tool("edit_file", {
                    "path": "calc.py", "old_str": "return a - b", "new_str": "return a + b",
                })),
                turn(tool("bash", {
                    "command": "python3 -c 'import calc; assert calc.add(2,3)==5; print(\"pass\")'",
                })),
                turn(text("Fixed and verified.")),
            ]),
        )
        agent.run("add() subtracts instead of adding")
        self.assertEqual("finished", agent.exit_reason)
        self.assertEqual(0, agent.usage.edit_failures)

        patch = extract_patch(self.env)
        self.assertIn("-    return a - b", patch)
        self.assertIn("+    return a + b", patch)

        # And the patch must actually apply to a clean checkout of the base commit.
        self.env.execute("git stash -q && git stash drop -q || git reset --hard -q HEAD")
        self.env.execute(f"cat > /tmp/p.diff <<'PATCH_EOF'\n{patch}\nPATCH_EOF")
        result = self.env.execute("git checkout -- . && git apply --check /tmp/p.diff")
        self.assertEqual(0, result["returncode"], result["output"])
