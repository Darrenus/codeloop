# codeloop

[![tests](https://github.com/Darrenus/codeloop/actions/workflows/tests.yml/badge.svg)](https://github.com/Darrenus/codeloop/actions/workflows/tests.yml)

A minimal, hackable terminal coding agent — written from scratch to study the parts
of a coding agent that actually matter: **context engineering, code-edit formats,
and sandboxing**.

The agent loop is ~110 lines. That is deliberate. An agent is an LLM, a loop, and
enough tokens; the engineering lives everywhere *except* the loop — in the model
and environment seams, the edit format, the approval policy, and the machinery
that makes a run reproducible.

> **Status.** The agent and its evaluation harness are built and tested (73 tests,
> CI on Python 3.9/3.11/3.13 plus a Docker integration job). **The SWE-bench
> ablation has not been run.** Every claim below is labelled as either *measured*
> — with the number and how it was obtained — or *reasoned*, meaning a design
> rationale that has not yet been tested. Nothing in this README is a benchmark
> result, because no benchmark has been run.

## Architecture

```
                 ┌────────────────────────────────────────────────┐
   task ───────► │  Agent.run()                                   │
                 │    messages ──► Model ──► stop_reason?          │
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
| [`agent.py`](src/codeloop/agent.py) | The loop, retry policy, and the `Usage` record every ablation compares. |
| [`model.py`](src/codeloop/model.py) | The model seam: Anthropic and OpenAI-compatible backends, provider presets, and the completion cache. |
| [`env.py`](src/codeloop/env.py) | The execution seam: local vs. Docker, output clipping, workspace path guard. |
| [`tools.py`](src/codeloop/tools.py) | Tool schemas and dispatch, parameterised by **edit format**. |
| [`patch.py`](src/codeloop/patch.py) | Working tree → submission diff, with build artefacts excluded. |
| [`cli.py`](src/codeloop/cli.py) | Terminal front-end and the interactive approval prompt. |
| [`eval/`](eval/) | SWE-bench batch runner, grader, and the A/B ablation driver. |

### Design decisions

*Measured* entries cite a number this repo actually produced. *Reasoned* entries
are design rationale that has not been tested — they are the reason the ablation
harness exists.

| Decision | Why | |
|---|---|---|
| Every tool is a shell command run through an `Environment` | One seam means the same agent drives your laptop and a per-instance container with zero branching in the agent or the tools. | reasoned |
| One canonical message format, translated per provider | Hardcoding a vendor ties the project's running cost to one price list and makes the loop untestable without a paid account. Both turned out to matter. | reasoned |
| Recorded trajectories replay by position, not by request hash | Most harness bugs are found *after* the completions were paid for, so reruns have to be free. A request-hash cache cannot deliver that: hosted models are not deterministic even at temperature 0 — four identical DeepSeek requests returned four differently-worded answers — so one differing token in step one makes every later request miss. Replaying by position sidesteps determinism entirely. | measured |
| The edit format is a swappable toolset, not a hardcoded tool | The hypothesis is that it is a large lever on token cost and edit-application failure rate. That is a claim, not a finding — making it a swappable variable is what would let it be tested. **Untested.** | reasoned |
| SEARCH/REPLACE `old_str` must match **exactly once** | An ambiguous match fails loudly instead of silently patching the wrong call site. | reasoned |
| Edit strings crossing the shell are base64-encoded | Code is full of `$`, backticks and quotes; encoding sidesteps every layer of quoting. A test edits `echo $(id) \`pwd\` "q"` in-container to prove it. | measured |
| Tool errors are returned to the model, not raised | A model that can read its own failure usually fixes it next turn; a traceback ends the run. | reasoned |
| Read-only tools bypass approval | Approval fatigue makes users hit "yes" reflexively, which defeats the whole mechanism. | reasoned |
| The container is long-lived (`sleep infinity` + `docker exec`) | The agent's third command depends on the state its second command left behind; one-shot `docker run` per tool call would throw that away. | reasoned |
| Output clipped head **and** tail | A flooded stdout carries its signal at the edges — the command echo at the top, the error at the bottom. | reasoned |
| Build artefacts are excluded from the submission patch | An agent that verifies its own fix leaves `.pyc` files behind; `git add -A` stages them, and the resulting binary hunks stop the patch applying to a clean checkout. Found by a test that runs `git apply --check`, not by inspection. | measured |
| Retries use full jitter; permanent errors fail instantly | Every worker in a batch hits the rate limit at the same moment, so synchronised backoff just re-collides. Providers overload 429 for both "too fast" and "account empty" — observed live from Gemini — so only the message tells them apart. | measured |
| The message list *is* the trajectory | No second representation to keep in sync, so runs are trivially replayable and gradeable. | reasoned |
| System prompt + tool schemas are cache-marked | They are byte-identical across every call in a trajectory. DeepSeek's automatic prefix cache was observed returning 384 cached input tokens on the second turn of a live run. | measured |

## Usage

```bash
pip install -e .
codeloop --list-providers                  # what is wired up, and which keys are set
codeloop -p deepseek "add a --verbose flag to the CLI and a test for it"
codeloop -p ollama -m qwen2.5-coder:7b "…"  # local, free, no key
```

Eleven providers ship as presets — Anthropic, DeepSeek, Qwen, GLM, Moonshot,
SiliconFlow, Groq, Cerebras, OpenRouter, Gemini, and any local Ollama or vLLM
server. Adding one is a single line. Keys come from the environment or the nearest
`.env`. See [`docs/providers.md`](docs/providers.md) for what the loop actually
requires of a model, and which options are free.

Flags: `-p/--provider`, `-m/--model`, `--edit-format {search_replace,whole_file}`,
`--max-steps`, `--replay TRAJECTORY`, `--cache-dir DIR`, `--docker-image IMAGE`, `--yolo`, `--trajectory PATH`.

Every run prints a usage footer: steps, input/output/cached tokens, edit success
ratio, wall time.

## What was actually measured

Four findings, all produced by running the thing rather than reasoning about it.
They are small, but they are the repo's only empirical claims and each one is
reproducible from the tests or the commit that fixed it.

| Finding | Number | How |
|---|---|---|
| `docker exec -w DIR` fails when `DIR` is absent — including the exec that would create it | — | Integration test against a fresh container; fixed by a bootstrap exec that omits `-w`. |
| An agent that verifies its own fix leaves `.pyc` files that `git add -A` stages, producing binary hunks that stop the patch applying | — | `git apply --check` in a container test; fixed by artefact pathspec exclusions in `patch.py`. |
| Providers return 429 both for rate limiting and for an empty account; backing off from the second is pure waste | — | Observed live from Gemini (`prepayment credits are depleted`); classified by message in `agent.py`. |
| Hosted models are not deterministic at temperature 0 | 4 identical DeepSeek requests → 4 different answers | Direct A/A probe. This is why replay is positional rather than request-hashed. |
| Replay costs nothing and reproduces the run | 16.0s / 9,428 tokens live → 0.6s / 0 tokens replayed, same patch | Smoke task recorded then replayed. |
| amd64 emulation penalty on Apple Silicon | 4.9s native arm64 → 10.4s under QEMU (≈2.1x) | Same 14-test container suite under both platforms. |

## Evaluation

Requires Docker and `pip install -e ".[eval]"`.

```bash
# one arm — --cache-dir makes every rerun free
python eval/run_swebench.py --split verified --n 30 -p deepseek \
    --cache-dir eval/cache --run-name sr-baseline
python eval/score.py --run-name sr-baseline

# the A/B: same 30 instances, same seed, one variable changed
python eval/ablate.py --n 30 --arms search_replace whole_file
```

SWE-bench images are `linux/amd64` only, so an Apple Silicon Mac runs them under
emulation at roughly 3-5x the wall time. [`docs/remote-runner.md`](docs/remote-runner.md)
covers running the benchmark on a native x86 box over SSH instead.

`ablate.py` prints a markdown table over resolve rate, empty-patch count, mean
steps, tokens per instance, **edit-failure rate**, and step-limit hits.

**The harness is built and unit-tested end to end against a scripted model, but
no benchmark run has been performed.** Both arms of the edit-format ablation are
implemented; what is missing is the spend and the machine time to run them.

Planned ablations, each holding the instance set, model and seed fixed:

1. **Edit format** — SEARCH/REPLACE vs. whole-file rewrite. *(both arms implemented, not run)*
2. **Repo map** — tree-sitter map on vs. off, on multi-file fixes. *(not built)*
3. **Compaction** — sliding window vs. summarisation, on long-horizon instances. *(not built)*

## Roadmap

- [x] **Stage 1** — agent loop, tool surface, approval layer, path-escape guard
- [x] **Stage 2** — `Environment` abstraction; per-instance Docker isolation; retry/abort policy
- [x] **Stage 2b** — provider-agnostic model seam, 11 presets, completion cache, trajectory replay; 73 tests
- [x] **Stage 3** — SWE-bench batch runner, grader, ablation driver, trajectory replay *(built, not run)*
- [ ] **Stage 4** — run the ablation and publish the table
- [ ] **Stage 5** — tree-sitter repo map, history compaction
- [ ] **Stage 6** — OS-level sandbox (Seatbelt / Landlock), MCP client, sub-agent delegation

Stage 5 has a concrete motivation from Stage 3: running the smoke task under
`--yolo`, the agent found `pytest` missing and ran `pip install --user pytest`
against the host. Correct behaviour for the task, and exactly what an OS-level
sandbox is for.

## Tests

No model is called by any test, so the suite is free to run.

```bash
python -m unittest discover -s tests -v      # all 73
python -m unittest tests.test_tools -v       # tool layer, temp dir, no Docker
python -m unittest tests.test_agent_loop -v  # the loop, against a scripted model
python -m unittest tests.test_model -v       # wire-format translation and the cache
python -m unittest tests.test_docker_env -v  # container seam, skipped without Docker
```

The agent's model client is injectable, so [`tests/fake_model.py`](tests/fake_model.py)
replays a scripted set of turns and the loop can be tested end to end with no API
key and no spend — usage accounting, the approval gate, edit-failure attribution,
step limits, parallel tool calls, and error recovery.

The integration suite rehearses what the harness does: a long-lived container
that keeps filesystem state across `docker exec` calls, the tool surface running
inside it, and `git diff` patch extraction. It defaults to the host architecture;
set `CODELOOP_TEST_PLATFORM=linux/amd64` to exercise the emulated path that the
real SWE-bench images require.

On an M5 Mac that emulated path costs about **2.1x** wall time on this suite
(4.9s native arm64 vs. 10.4s under QEMU) — cheap here because the suite is
process-spawn bound, but the reason real benchmark runs belong on an x86 host.

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
