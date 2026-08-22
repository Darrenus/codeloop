"""Execution environments.

Every tool ultimately becomes a shell command run inside an `Environment`.
That single seam is what lets the same agent drive the local machine during
development and a per-instance Docker container during evaluation, with no
branching inside the agent or the tools.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Optional

MAX_OUTPUT_CHARS = 30_000


def clip(text: str) -> str:
    """Keep the head and the tail. A flooded stdout carries its signal at the
    edges -- the command echo at the top, the error at the bottom."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n\n... [{omitted} characters omitted] ...\n\n{text[-half:]}"


class Environment:
    cwd: str = "."

    def execute(self, command: str, timeout: int = 120) -> dict:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class LocalEnvironment(Environment):
    """Runs commands on this machine, confined to a workspace root."""

    def __init__(self, cwd: Optional[str] = None, env: Optional[dict] = None):
        self.cwd = str(Path(cwd or os.getcwd()).resolve())
        self.env = env or {}

    def execute(self, command: str, timeout: int = 120) -> dict:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **self.env},
            )
        except subprocess.TimeoutExpired:
            return {"output": f"[timed out after {timeout}s]", "returncode": 124}
        output = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        return {"output": clip(output), "returncode": proc.returncode}

    def guard_path(self, path: str) -> None:
        """Refuse paths that escape the workspace root.

        Only meaningful locally -- inside a per-instance container the whole
        filesystem is disposable, so there is nothing to protect."""
        root = Path(self.cwd)
        target = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path escapes the workspace root: {path}")


class DockerEnvironment(Environment):
    """Runs commands inside a long-lived container.

    Started detached with `sleep infinity` and driven with `docker exec`, so the
    container keeps its filesystem state across the whole trajectory -- which is
    the entire point when the agent's third command depends on its second.
    """

    def __init__(self, image: str, cwd: str = "/testbed", platform: str = "linux/amd64"):
        self.image = image
        self.cwd = cwd
        self.container = f"codeloop-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--platform", platform,
                "--name", self.container,
                image, "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
        # `docker exec -w DIR` fails outright when DIR does not exist, and that
        # includes the exec that would have created it. Bootstrap it without -w.
        subprocess.run(
            ["docker", "exec", self.container, "bash", "-lc", f"mkdir -p {shlex.quote(cwd)}"],
            capture_output=True,
        )

    def execute(self, command: str, timeout: int = 120) -> dict:
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", self.cwd, self.container, "bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"output": f"[timed out after {timeout}s]", "returncode": 124}
        output = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
        return {"output": clip(output), "returncode": proc.returncode}

    def guard_path(self, path: str) -> None:
        return None

    def cleanup(self) -> None:
        subprocess.run(["docker", "kill", self.container], capture_output=True)
