# AI Data-Analysis Agent (LangGraph)

An agent that turns a **dataset + a question** into a **Markdown report** with cleaning
notes, computed tables, charts and a written interpretation.

It implements the workflow in [ai_agent_01_workflow.png](ai_agent_01_workflow.png):

```
                    ┌─────────────────┐
        ┌──────────▶│  Data Cleaning  │──────────┐
        │           └─────────────────┘          ▼
┌───────────────┐                        ┌─────────────────┐        ┌──────────────┐
│  Input Data   │                        │  Data Analysis  │───────▶│ Final Report │
│  + User Prompt│                        │  using Python   │        │    Ready     │
└───────────────┘                        └─────────────────┘        └──────────────┘
        │           ┌─────────────────┐    ▲         │                      ▲
        └──────────▶│ LLM Integration │────┘         │                      │
                    └─────────────────┘◀─────────────┘                      │
                             └──────────────────────────────────────────────┘
```

The LLM plans, writes Python, repairs its own tracebacks and narrates the findings.
**Every number in the report comes from pandas actually running**, never from the model.

**Two interchangeable backends:** local models via **Ollama**, or the **OpenAI API**.

📖 Full walkthrough of the design and how to build it: **[GUIDE.md](GUIDE.md)**

---

## Quick start

The `.venv` is already created and populated.

```powershell
# 1. Pick a backend in .env  (LLM_PROVIDER=ollama | openai)

# 2. Check it answers
.venv\Scripts\python.exe -m agent --check

# 3. Run an analysis
.venv\Scripts\python.exe -m agent `
    -d data\sample_sales.csv `
    -p "Which region generates the most revenue, and how do ratings compare across categories?"
```

### Backend A — local, via Ollama

```powershell
ollama serve                     # in its own terminal
ollama pull qwen2.5-coder:7b     # recommended; code-tuned
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
```

### Backend B — OpenAI API

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Either can be overridden per run without touching `.env`:

```powershell
.venv\Scripts\python.exe -m agent -d data\sample_sales.csv -p "..." --provider openai --model gpt-4o-mini
```

---

## CLI

| Flag | Meaning |
|---|---|
| `-d`, `--data` | Dataset: `.csv` `.tsv` `.json` `.xlsx` `.parquet` |
| `-p`, `--prompt` | The question to answer |
| `--provider` | `ollama` or `openai`, overriding `.env` |
| `--model` | Model name, overriding `.env` |
| `--retries` | Repair attempts before giving up (default 3) |
| `--check` | Probe the backend and exit |
| `--show-report` | Render the finished report in the terminal |

---

## Output

Each run writes a self-contained, auditable directory:

```
runs/<timestamp>_<id>/
├── report.md                   ← the deliverable
├── figures/*.png
├── clean.csv / clean.pkl       ← the cleaned dataset
├── raw_profile.json
├── clean_profile.json
├── generated_attempt_N.py      ← exactly what the LLM wrote, each attempt
├── analysis_attempt_N.py       ← the harnessed script that ran
└── result.json
```

---

## Project layout

| Path | Role |
|---|---|
| [agent/graph.py](agent/graph.py) | LangGraph wiring: fan-out, join, the repair loop |
| [agent/state.py](agent/state.py) | Shared state channels and their reducers |
| [agent/llm.py](agent/llm.py) | Dual backend factory + forgiving output parsers |
| [agent/config.py](agent/config.py) | `.env`-driven settings and validation |
| [agent/profiling.py](agent/profiling.py) | Loading, profiling, deterministic cleaning |
| [agent/sandbox.py](agent/sandbox.py) | Subprocess execution of generated code |
| [agent/prompts.py](agent/prompts.py) | The four prompt templates |
| [agent/nodes/](agent/nodes/) | One module per node in the diagram |
| [agent/cli.py](agent/cli.py) | Terminal entry point |
| [tests/test_graph_offline.py](tests/test_graph_offline.py) | End-to-end test with a scripted fake LLM |

---

## Test without a model or a key

```powershell
.venv\Scripts\python.exe tests\test_graph_offline.py
```

A scripted `FakeLLM` returns broken code on the first attempt and correct code on the
repair, so the run exercises the fan-out, the reducers, the sandbox, the repair loop
and the report — in about two seconds, with no network.

---

## Reliability notes

* **Repair loop.** Execution failures go back to the model with the verbatim traceback.
  In testing, a 3B local model failed twice and succeeded on the third attempt.
* **Partial results survive.** If the code crashes halfway, whatever it already computed
  is kept and reported rather than discarded.
* **Auto-capture.** If the model computes aggregates but forgets to assign them into
  `RESULTS`, the harness adopts them from the namespace.
* **Sandbox scope.** The deny-list and subprocess isolation guard against *accidents*
  (a hallucinated `import os`, a runaway loop), not against deliberately hostile code.
  For untrusted input, run the subprocess in a container.
