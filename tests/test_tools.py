"""Tool-layer tests. No API key and no Docker required -- everything here runs
against LocalEnvironment in a temp directory."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeloop import tools  # noqa: E402
from codeloop.env import LocalEnvironment, clip  # noqa: E402


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = LocalEnvironment(cwd=self.tmp.name)
        Path(self.tmp.name, "a.py").write_text("def f():\n    return 1\n")
        Path(self.tmp.name, "sub").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_file_has_line_numbers(self):
        self.assertIn("1\tdef f():", tools.read_file(self.env, "a.py"))

    def test_read_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            tools.read_file(self.env, "nope.py")

    def test_list_files_marks_directories(self):
        out = tools.list_files(self.env, ".")
        self.assertIn("sub/", out)
        self.assertIn("a.py", out)

    def test_grep_reports_no_matches(self):
        self.assertEqual("(no matches)", tools.grep(self.env, "zzz").strip())

    def test_grep_finds_line(self):
        self.assertIn("return 1", tools.grep(self.env, "return"))

    def test_edit_replaces_unique_match(self):
        tools.edit_file(self.env, "a.py", "return 1", "return 2")
        self.assertIn("return 2", Path(self.tmp.name, "a.py").read_text())

    def test_edit_rejects_ambiguous_match(self):
        Path(self.tmp.name, "b.py").write_text("x = 1\nx = 1\n")
        with self.assertRaises(RuntimeError) as ctx:
            tools.edit_file(self.env, "b.py", "x = 1", "x = 2")
        self.assertIn("appears 2 times", str(ctx.exception))

    def test_edit_rejects_missing_match(self):
        with self.assertRaises(RuntimeError):
            tools.edit_file(self.env, "a.py", "not there", "x")

    def test_edit_creates_file_when_old_str_empty(self):
        tools.edit_file(self.env, "new/deep.py", "", "hello\n")
        self.assertEqual("hello\n", Path(self.tmp.name, "new/deep.py").read_text())

    def test_edit_survives_shell_metacharacters(self):
        Path(self.tmp.name, "c.sh").write_text("echo 'a b' $(x) \"q\"\n")
        tools.edit_file(self.env, "c.sh", "$(x)", "`y`")
        self.assertIn("`y`", Path(self.tmp.name, "c.sh").read_text())

    def test_write_file_replaces_whole_contents(self):
        tools.write_file(self.env, "a.py", "# gone\n")
        self.assertEqual("# gone\n", Path(self.tmp.name, "a.py").read_text())

    def test_bash_reports_exit_code(self):
        self.assertIn("[exit 3]", tools.bash(self.env, "exit 3"))

    def test_path_escape_is_refused(self):
        with self.assertRaises(ValueError):
            tools.read_file(self.env, "../../etc/passwd")

    def test_clip_keeps_head_and_tail(self):
        out = clip("A" * 10 + "B" * 100_000 + "C" * 10)
        self.assertTrue(out.startswith("AAAA"))
        self.assertTrue(out.endswith("CCCC"))
        self.assertIn("characters omitted", out)


class ToolsetTests(unittest.TestCase):
    def test_search_replace_toolset(self):
        registry, schemas = tools.get_toolset("search_replace")
        self.assertIn("edit_file", registry)
        self.assertNotIn("write_file", registry)
        self.assertEqual({s["name"] for s in schemas}, set(registry))

    def test_whole_file_toolset(self):
        registry, schemas = tools.get_toolset("whole_file")
        self.assertIn("write_file", registry)
        self.assertNotIn("edit_file", registry)
        self.assertEqual({s["name"] for s in schemas}, set(registry))

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            tools.get_toolset("unified_diff")


if __name__ == "__main__":
    unittest.main()
