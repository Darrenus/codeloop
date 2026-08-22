# codeloop

A minimal, hackable terminal coding agent — written from scratch to study the parts
of a coding agent that actually matter: **context engineering, code-edit formats,
and sandboxing**.

The agent loop itself is ~90 lines. That is deliberate. An agent is an LLM, a loop,
and enough tokens; the engineering lives everywhere *except* the loop, and this repo
is a lab for measuring that claim rather than asserting it.

> **Status: work in progress.** Stage 1 (loop + tools + approval) is implemented and
> runnable. Stages 2–4 and all benchmark numbers below are not done yet — this
> section will be updated with real measurements as they land, and nothing is
> reported here that has not actually been run.

## Architecture

```
                 ┌──────────────────────────────────────────┐
   task ───────► │  Agent.run()                             │
                 │    messages ──► Claude ──► stop_reason?   │
                 │        ▲                    │             │
                 │        │              tool_use            │
                 │        │                    ▼             │
                 │        │        ┌───────────────────────┐ │
                 │        └────────┤ approval → dispatch   │ │
                 │   tool_result   └───────────┬───────────┘ │
                 └──────────────────────────────┼────────────┘
                                                ▼
                    read_file · list_files · grep · edit_file · bash
```

- **`src/codeloop/agent.py`** — the loop. Linear message history; the trajectory
  *is* the message list, which makes replay and evaluation trivial.
- **`src/codeloop/tools.py`** — tool schemas, dispatch, output clipping, and
  workspace-root path-escape checks.
- **`src/codeloop/cli.py`** — terminal front-end and the interactive approval prompt.

### Design decisions so far

| Decision | Why |
|---|---|
| SEARCH/REPLACE edits, not whole-file rewrites | Cost scales with the size of the change, not the size of the file. |
| `old_str` must match **exactly once** | An ambiguous match fails loudly instead of silently patching the wrong call site. |
| Errors are returned to the model, not raised | The model can read the failure and retry; a traceback ends the run. |
| Read-only tools bypass approval | Approval fatigue makes users hit "yes" reflexively, which defeats the point. |
| Tool output clipped head+tail | Preserves the parts that carry signal (the command echo and the error) when a command floods stdout. |

## Usage

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
codeloop "add a --verbose flag to the CLI and a test for it"
```

Flags: `--model`, `--max-steps`, `--yolo` (skip write approval), `--trajectory PATH`.

## Roadmap

- [x] **Stage 1** — agent loop, five tools, approval layer, path-escape guard
- [ ] **Stage 2** — OS-level sandbox (macOS Seatbelt / Linux Landlock), command allowlist
- [ ] **Stage 3** — context engineering: tree-sitter repo map, history compaction, prompt caching
- [ ] **Stage 4** — SWE-bench Verified harness + A/B ablations (edit format, repo map on/off)
- [ ] **Stage 5** — MCP client, sub-agent delegation for read-only exploration

## Prior art

Built while reading, and indebted to:

- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — the argument that a
  ~100-line agent can score >74% on SWE-bench Verified; also the source of the
  benchmark harness design this repo will follow.
- [Thorsten Ball, *How to Build an Agent*](https://ampcode.com/notes/how-to-build-an-agent)
  — the <400-line skeleton.
- [aider](https://github.com/Aider-AI/aider) — repo map and SEARCH/REPLACE edit format.
- [openai/codex](https://github.com/openai/codex) — sandbox and approval-mode design.

## License

MIT
