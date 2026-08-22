"""Tool definitions and dispatch.

Each tool is a plain function plus a JSON schema. The schema list is what we
send to the model; the registry is what we execute against. Keeping the two
next to each other makes it obvious when they drift apart.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAX_OUTPUT_CHARS = 30_000


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"


def _resolve(path: str) -> Path:
    """Resolve a user-supplied path and refuse to escape the workspace root."""
    root = Path.cwd().resolve()
    target = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"path escapes the workspace root: {path}")
    return target


def read_file(path: str) -> str:
    target = _resolve(path)
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return _clip("\n".join(f"{i:6d}\t{line}" for i, line in enumerate(lines, 1)))


def list_files(path: str = ".") -> str:
    target = _resolve(path)
    entries = sorted(
        p for p in target.iterdir() if not p.name.startswith(".")
    )
    return _clip("\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries))


def grep(pattern: str, path: str = ".") -> str:
    target = _resolve(path)
    proc = subprocess.run(
        ["grep", "-rn", "--exclude-dir=.git", "-e", pattern, str(target)],
        capture_output=True,
        text=True,
    )
    return _clip(proc.stdout or "(no matches)")


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """SEARCH/REPLACE edit. An empty `old_str` on a missing file creates it.

    We require `old_str` to appear exactly once so an ambiguous match fails
    loudly instead of silently patching the wrong call site.
    """
    target = _resolve(path)
    if not target.exists():
        if old_str:
            raise FileNotFoundError(f"{path} does not exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_str, encoding="utf-8")
        return f"created {path}"

    content = target.read_text(encoding="utf-8")
    occurrences = content.count(old_str)
    if occurrences == 0:
        raise ValueError(f"old_str not found in {path}")
    if occurrences > 1:
        raise ValueError(
            f"old_str appears {occurrences} times in {path}; include more surrounding context"
        )
    target.write_text(content.replace(old_str, new_str), encoding="utf-8")
    return f"edited {path}"


def bash(command: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout
    )
    parts = []
    if proc.stdout:
        parts.append(proc.stdout)
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr}")
    parts.append(f"[exit {proc.returncode}]")
    return _clip("\n".join(parts))


REGISTRY = {
    "read_file": read_file,
    "list_files": list_files,
    "grep": grep,
    "edit_file": edit_file,
    "bash": bash,
}

# Tools that never mutate state — the approval layer waves these through.
READ_ONLY = {"read_file", "list_files", "grep"}

SCHEMAS = [
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
        "description": "List entries in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "grep",
        "description": "Recursively search for a pattern and return matching lines.",
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
        "name": "edit_file",
        "description": (
            "Replace old_str with new_str in a file. old_str must match exactly once. "
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
    },
    {
        "name": "bash",
        "description": "Run a shell command in the workspace and return stdout, stderr and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]
