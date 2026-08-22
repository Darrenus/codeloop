"""Tool definitions and dispatch.

Every tool is expressed as a shell command executed through an `Environment`,
so the same tool surface works unchanged against the local machine and against
a Docker container.

The tool *set* is parameterised by edit format. That is deliberate: the format
in which a model is asked to express a code change is the single biggest lever
on both token cost and edit-application failure rate, and `eval/` A/B tests it.
"""
from __future__ import annotations

import base64
import shlex
from typing import Callable, Dict, List, Tuple

from .env import Environment

# --------------------------------------------------------------------------
# The edit primitive runs inside the environment, so it works identically in a
# container. Passing the strings as base64 sidesteps every layer of shell and
# heredoc quoting.
# --------------------------------------------------------------------------
_EDIT_SCRIPT = r"""
import base64, sys, pathlib
path, old_b64, new_b64 = sys.argv[1], sys.argv[2], sys.argv[3]
old = base64.b64decode(old_b64).decode()
new = base64.b64decode(new_b64).decode()
p = pathlib.Path(path)
if not p.exists():
    if old:
        print("ERROR: %s does not exist" % path); sys.exit(1)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new); print("created %s" % path); sys.exit(0)
content = p.read_text()
n = content.count(old)
if n == 0:
    print("ERROR: old_str not found in %s" % path); sys.exit(1)
if n > 1:
    print("ERROR: old_str appears %d times in %s; include more surrounding context" % (n, path))
    sys.exit(1)
p.write_text(content.replace(old, new, 1))
print("edited %s" % path)
"""

_WRITE_SCRIPT = r"""
import base64, sys, pathlib
path, content_b64 = sys.argv[1], sys.argv[2]
p = pathlib.Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(base64.b64decode(content_b64).decode())
print("wrote %s" % path)
"""


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _python(env: Environment, script: str, *args: str) -> str:
    cmd = "python3 -c {} {}".format(
        shlex.quote(script), " ".join(shlex.quote(a) for a in args)
    )
    result = env.execute(cmd)
    if result["returncode"] != 0:
        raise RuntimeError(result["output"].strip() or "command failed")
    return result["output"].strip()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def read_file(env: Environment, path: str) -> str:
    env.guard_path(path)
    result = env.execute(f"cat -n {shlex.quote(path)}")
    if result["returncode"] != 0:
        raise FileNotFoundError(result["output"].strip())
    return result["output"]


def list_files(env: Environment, path: str = ".") -> str:
    env.guard_path(path)
    result = env.execute(f"ls -A -p {shlex.quote(path)}")
    if result["returncode"] != 0:
        raise FileNotFoundError(result["output"].strip())
    return result["output"]


def grep(env: Environment, pattern: str, path: str = ".") -> str:
    env.guard_path(path)
    result = env.execute(
        f"grep -rn --exclude-dir=.git -e {shlex.quote(pattern)} {shlex.quote(path)}"
    )
    # grep exits 1 on "no matches", which is not an error worth raising.
    return result["output"] or "(no matches)"


def edit_file(env: Environment, path: str, old_str: str, new_str: str) -> str:
    """SEARCH/REPLACE edit. `old_str` must match exactly once."""
    env.guard_path(path)
    return _python(env, _EDIT_SCRIPT, path, _b64(old_str), _b64(new_str))


def write_file(env: Environment, path: str, content: str) -> str:
    """Whole-file rewrite. The control arm of the edit-format ablation."""
    env.guard_path(path)
    return _python(env, _WRITE_SCRIPT, path, _b64(content))


def bash(env: Environment, command: str) -> str:
    result = env.execute(command)
    return f"{result['output']}\n[exit {result['returncode']}]"


# --------------------------------------------------------------------------
# Registry, schemas, and the edit-format toolsets
# --------------------------------------------------------------------------
_BASE_REGISTRY: Dict[str, Callable] = {
    "read_file": read_file,
    "list_files": list_files,
    "grep": grep,
    "bash": bash,
}

READ_ONLY = {"read_file", "list_files", "grep"}

_BASE_SCHEMAS: List[dict] = [
    {
        "name": "read_file",
        "description": "Read a file's contents with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": "List entries in a directory. Directories are suffixed with '/'.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "grep",
        "description": "Recursively search for a pattern and return matching lines with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command and return stdout, stderr and the exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

_EDIT_SCHEMA = {
    "name": "edit_file",
    "description": (
        "Replace old_str with new_str in a file. old_str must appear exactly once, "
        "so include enough surrounding context to make it unique. "
        "Pass an empty old_str to create a new file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
        "required": ["path", "old_str", "new_str"],
    },
}

_WRITE_SCHEMA = {
    "name": "write_file",
    "description": (
        "Write the complete new contents of a file, replacing whatever is there. "
        "You must supply the entire file, not a fragment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}

EDIT_FORMATS = ("search_replace", "whole_file")


def get_toolset(edit_format: str = "search_replace") -> Tuple[Dict[str, Callable], List[dict]]:
    """Return (registry, schemas) for an edit format."""
    if edit_format not in EDIT_FORMATS:
        raise ValueError(f"unknown edit format {edit_format!r}; expected one of {EDIT_FORMATS}")
    registry = dict(_BASE_REGISTRY)
    schemas = list(_BASE_SCHEMAS)
    if edit_format == "search_replace":
        registry["edit_file"] = edit_file
        schemas.append(_EDIT_SCHEMA)
    else:
        registry["write_file"] = write_file
        schemas.append(_WRITE_SCHEMA)
    return registry, schemas
