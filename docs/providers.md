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

## Never paying twice: `--cache-dir`

Every completion is keyed by a hash of the model, system prompt, message history
and tool schemas, then written to disk. An identical request replays from the
cache instead of hitting the network.

```bash
python eval/run_swebench.py --n 5 --cache-dir eval/cache --run-name pilot
python eval/run_swebench.py --n 5 --cache-dir eval/cache --run-name pilot --resume
```

The first run costs money; every rerun is free. This matters more than it sounds:
most harness bugs — the `.pyc` in the patch, a metric computed wrong, a broken
ablation table — are found *after* the completions were paid for, and without a
cache each fix means paying again. It also makes a published result exactly
reproducible from the committed cache.

Replayed replies report zero usage, so a cached rerun cannot inflate the cost
column in the metrics.
