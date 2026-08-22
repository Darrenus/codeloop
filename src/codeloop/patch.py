"""Turning a working tree into a submission patch.

Sounds like a one-liner, and `git add -A && git diff --cached` almost is -- until
the agent runs the test suite to verify its own fix and git faithfully stages
every `.pyc` it produced. The resulting patch carries binary hunks and no longer
applies to a clean checkout, so the instance is scored as a failure for reasons
that have nothing to do with the model.
"""
from __future__ import annotations

import shlex

from .env import Environment

#: Build and cache artefacts a test run leaves behind. Excluded from both the
#: staging step and the diff, so they can neither enter the patch nor dirty it.
ARTIFACT_EXCLUDES = (
    "**/__pycache__/**",
    "**/*.py[cod]",
    "**/*.so",
    "**/*.egg-info/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.ruff_cache/**",
    "**/.tox/**",
    "**/.coverage",
    "**/*.orig",
    "**/*.rej",
)


def _pathspec() -> str:
    return " ".join(shlex.quote(f":(exclude){p}") for p in ARTIFACT_EXCLUDES)


def extract_patch(env: Environment) -> str:
    """Stage the agent's work and return it as a unified diff.

    Staging is what makes newly created files show up in the diff at all; the
    exclusions are what keep the agent's own test run out of it.
    """
    spec = _pathspec()
    env.execute(f"git add -A -- . {spec}")
    return env.execute(f"git diff --cached -- . {spec}")["output"]
