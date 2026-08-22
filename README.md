# codeloop

[![tests](https://github.com/Darrenus/codeloop/actions/workflows/tests.yml/badge.svg)](https://github.com/Darrenus/codeloop/actions/workflows/tests.yml)

A minimal, hackable terminal coding agent — written from scratch to study the parts
of a coding agent that actually matter: **context engineering, code-edit formats,
and sandboxing**.

The agent loop is ~110 lines. That is deliberate. An agent is an LLM, a loop, and
enough tokens; the engineering lives everywhere *except* the loop. So this repo
pairs a small agent with a SWE-bench harness that **measures** that claim instead
of asserting it.

> **Status.** Stages 1–2 are implemented and tested. The Stage 4 harness is written
> and runnable but **has not been run yet — there are no benchmark numbers in this
> README, and none will be added that were not actually measured.** Results land in
> `eval/results/` as raw trajectories alongside the summary table, so every number
> can be re-derived.

## Architecture

```
                 ┌────────────────────────────────────────────────┐
   task ───────► │  Agent.run()                                   │
                 │    messages ──► Claude ──► stop_reason?         │
                 │        ▲                      │                 │
                 │        │                tool_use                │
                 │        │                      ▼                 │
                 │        │        ┌───────────────────────────┐   │
                 │        └────────┤ approval → dispatch       │   │
                 │   tool_result   └─────────────┬─────────────┘   │
                 └──────────────────────────────┼─────────────────┘
                                                ▼
                                       Environment.execute()
                                       ╱                  ╲
                            LocalEnvironment        DockerEnvironment
                          (dev, path-guarded)    (one container per instance)
```

| Module | Responsibility |
|---|---|
| [`agent.py`](src/codeloop/agent.py) | The loop, prompt caching, and the `Usage` record every ablation compares. |
| [`env.py`](src/codeloop/env.py) | The execution seam: local vs. Docker, output clipping, workspace path guard. |
| [`tools.py`](src/codeloop/tools.py) | Tool schemas and dispatch, parameterised by **edit format**. |
| [`cli.py`](src/codeloop/cli.py) | Terminal front-end and the interactive approval prompt. |
| [`eval/`](eval/) | SWE-bench batch runner, grader, and the A/B ablation driver. |

### Design decisions

| Decision | Why |
|---|---|
| Every tool is a shell command run through an `Environment` | One seam means the same agent drives your laptop and a per-instance container with zero branching in the agent or the tools. |
| The edit format is a swappable toolset, not a hardcoded tool | It is the single biggest lever on token cost and edit-failure rate, so it has to be an experimental variable. |
| SEARCH/REPLACE `old_str` must match **exactly once** | An ambiguous match fails loudly instead of silently patching the wrong call site. |
| Edit strings crossing the shell are base64-encoded | Code is full of `$`, backticks and quotes; encoding sidesteps every layer of quoting. Tested against exactly that. |
| Tool errors are returned to the model, not raised | A model that can read its own failure usually fixes it next turn; a traceback ends the run. |
| Read-only tools bypass approval | Approval fatigue makes users hit "yes" reflexively, which defeats the whole mechanism. |
| Output clipped head **and** tail | A flooded stdout carries its signal at the edges — the command echo at the top, the error at the bottom. |
| The message list *is* the trajectory | No second representation to keep in sync, so runs are trivially replayable and gradeable. |
| System prompt + tool schemas are cache-marked | They are byte-identical across ~50 calls per trajectory. |

## Usage

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
codeloop "add a --verbose flag to the CLI and a test for it"
```

Flags: `--model`, `--edit-format {search_replace,whole_file}`, `--max-steps`,
`--docker-image IMAGE`, `--yolo`, `--trajectory PATH`.

Every run prints a usage footer: steps, input/output/cached tokens, edit success
ratio, wall time.

## Evaluation

Requires Docker and `pip install -e ".[eval]"`.

```bash
# one arm
python eval/run_swebench.py --split verified --n 30 --run-name sr-baseline
python eval/score.py --run-name sr-baseline

# the A/B: same 30 instances, same seed, one variable changed
python eval/ablate.py --n 30 --arms search_replace whole_file
```

`ablate.py` prints a markdown table over resolve rate, empty-patch count, mean
steps, tokens per instance, **edit-failure rate**, and step-limit hits.

Planned ablations, each holding the instance set, model and seed fixed:

1. **Edit format** — SEARCH/REPLACE vs. whole-file rewrite. *(implemented)*
2. **Repo map** — tree-sitter map on vs. off, on multi-file fixes. *(Stage 3)*
3. **Compaction** — sliding window vs. summarisation, on long-horizon instances. *(Stage 3)*

## Roadmap

- [x] **Stage 1** — agent loop, tool surface, approval layer, path-escape guard
- [x] **Stage 2** — `Environment` abstraction; per-instance Docker isolation
- [ ] **Stage 3** — tree-sitter repo map, history compaction, richer caching
- [ ] **Stage 4** — run the harness; publish the edit-format ablation
- [ ] **Stage 5** — OS-level sandbox (Seatbelt / Landlock), MCP client, sub-agent delegation

## Tests

No API key and no Docker needed — the tool layer is tested against a temp directory.

```bash
python -m unittest discover -s tests -v
```

## Prior art

Built while reading, and indebted to:

- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — the argument that a
  ~100-line agent can score >74% on SWE-bench Verified; the batch-harness design here
  follows its structure (per-instance image naming, resumable predictions file).
- [Thorsten Ball, *How to Build an Agent*](https://ampcode.com/notes/how-to-build-an-agent) — the <400-line skeleton.
- [aider](https://github.com/Aider-AI/aider) — repo map and the SEARCH/REPLACE edit format.
- [openai/codex](https://github.com/openai/codex) — sandbox and approval-mode design.

## License

MIT
