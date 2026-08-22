# Choosing a model provider

codeloop talks to any provider through [`model.py`](../src/codeloop/model.py).
Exactly one backend speaks Anthropic's protocol; everything else speaks the
OpenAI chat-completions protocol, which by now is the lingua franca — including
for local servers. Adding a provider is one line in `PROVIDERS`.

```bash
codeloop --list-providers          # what is configured, and which keys are set
codeloop -p deepseek "fix the failing test"
codeloop -p ollama -m qwen2.5-coder:7b "fix the failing test"
```

Keys are read from the environment or from the nearest `.env` walking up from the
working directory. `.env` is gitignored.

## What the agent actually needs

Not every model can drive this loop. The requirements are narrow but strict:

- **Native tool calling.** The agent never parses tool calls out of prose. A model
  without function-calling support will appear to work and then do nothing.
- **A context window of 32k or more.** Trajectories accumulate file contents and
  command output; 8k models hit the wall within a few steps.
- **Reasonable instruction adherence.** Weak models burn steps re-reading files
  they already read, which shows up directly in the step and token columns.

Models below roughly 7B rarely clear the second and third bars. That is itself a
measurable result, and a legitimate thing for this repo to report.

## Free and low-cost options

Rate limits and free quotas move constantly — check the provider before planning
a run. This table records what each preset points at, not a promise about price.

| Preset | Key | Notes |
|---|---|---|
| `gemini` | `GEMINI_API_KEY` | Free tier via AI Studio, no card. Large context. The OpenAI-compatible endpoint supports tool calling. |
| `groq` | `GROQ_API_KEY` | Free tier, very fast. Daily request cap, so a batch run needs low `--workers`. |
| `cerebras` | `CEREBRAS_API_KEY` | Free tier, fast. |
| `openrouter` | `OPENROUTER_API_KEY` | Aggregates many providers; several models carry a `:free` suffix. Tool-calling support varies **by model** — verify before a run. |
| `ollama` | none | Local, free, no key, no network. See below. |
| `deepseek` | `DEEPSEEK_API_KEY` | Paid but very cheap, and settles in RMB via Alipay/WeChat. Off-peak pricing is roughly half. |
| `qwen` | `DASHSCOPE_API_KEY` | Alibaba DashScope. New accounts get a free token grant; RMB billing. |
| `glm` | `ZHIPU_API_KEY` | Zhipu. `glm-4-flash` has historically been free. RMB billing. |
| `moonshot` | `MOONSHOT_API_KEY` | RMB billing. |
| `siliconflow` | `SILICONFLOW_API_KEY` | Hosts open-weight coding models including Qwen Coder. RMB billing. |
| `anthropic` | `ANTHROPIC_API_KEY` | Requires an international card. Note that a Claude Pro/Max subscription does **not** fund API usage. |

## Running locally with Ollama

Free, offline, and enough for developing the harness. On a laptop with a 4 GB
GPU, a 7B model quantised to Q4 mostly fits in VRAM with the remainder on CPU:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
codeloop -p ollama -m qwen2.5-coder:7b --yolo "fix the failing test"
```

Expect it to be slow and to score poorly on real benchmark instances. That is the
correct expectation, and it is still useful: the harness, the tool surface, the
patch extraction and the metrics are all exercised for free, and only the final
measurement run needs a stronger model.

## Never paying twice: `--replay` and `--cache-dir`

Two mechanisms, and the difference between them matters.

### `--replay` — the guaranteed-free path

Record a run once, then replay its turns by position:

```bash
codeloop -p deepseek --trajectory run.json "fix the failing test"   # costs money once
codeloop --replay run.json "ignored"                                 # free, forever
```

For the benchmark, `--replay-dir` points at a previous run's `trajectories/`
directory and replays every instance:

```bash
python eval/run_swebench.py --n 30 -p deepseek --run-name v1
python eval/run_swebench.py --n 30 --run-name v1-refixed \
    --replay-dir eval/results/v1/trajectories
```

Measured on the smoke case: the live run took 16.0s and 9,428 tokens; the replay
took 0.6s and zero, and produced the same patch.

This is what makes harness iteration affordable. Most harness bugs — the `.pyc`
in the patch, a metric computed wrong, a broken ablation table — are only found
*after* the completions were paid for. Replay re-exercises the environment, the
tools, patch extraction and the metrics against a recording, as many times as it
takes. What it cannot tell you is how the model would have *reacted* to a change,
since the replies are fixed.

### `--cache-dir` — best effort, not a guarantee

Completions are also keyed by a hash of model, temperature, system prompt,
history and tool schemas, and served from disk on an identical request.

It helps less than it looks like it should, and the reason is worth knowing:
**hosted models are not deterministic even at temperature 0.** Four identical
requests to DeepSeek returned four differently-worded answers; batched
mixture-of-experts routing does not reproduce bit for bit. One differing token
in step one changes the history for step two, and every request after it misses.

So the cache reliably saves the *first* request of a repeated run and any
genuinely identical prefix, and little else. `codeloop` still defaults to
temperature 0 — reproducibility is worth asking for even where it is not
granted — but replay is what actually delivers free reruns.
