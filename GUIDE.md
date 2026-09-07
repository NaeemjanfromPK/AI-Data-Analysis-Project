# Building the Data-Analysis Agent with LangGraph

A step-by-step guide to the agent implemented in this workspace. It follows the
workflow diagram exactly: **Input → (Data Cleaning ‖ LLM Integration) → Data
Analysis in Python ⇄ LLM Integration → Final Report**.

---

## 1. What we are building

An agent that takes a **dataset** plus a **natural-language question** and returns a
**Markdown report** containing cleaning notes, an analysis plan, computed tables,
charts and a written interpretation.

The design principle throughout: **the LLM writes code, it never touches the data.**
Numbers come from pandas running in a real interpreter; the model only plans, writes
Python, repairs its own errors and narrates the results. That is what makes the output
reproducible and auditable — every run leaves the exact script it executed on disk.

### The five nodes

| Node | Diagram box | Uses an LLM? | Job |
|---|---|---|---|
| `ingest` | Input Data + User Prompt | no | Load csv/xlsx/json/parquet, profile it, create the run directory |
| `data_cleaning` | Data Cleaning | no | Deterministic, explainable cleaning; writes `clean.pkl` |
| `llm_integration` | LLM Integration | **yes** | Four stages: `plan`, `codegen`, `repair`, `interpret` |
| `data_analysis` | Data Analysis using Python | no | Executes generated code in a sandboxed subprocess; decides where to go next |
| `final_report` | Final Report Ready | no | Assembles the Markdown deliverable |

### The edges, and why they exist

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

* **Fan-out** — cleaning is CPU work, planning is a network round-trip to the model.
  They do not depend on each other, so LangGraph runs them in the same superstep.
* **Join** — `data_analysis` has two inbound edges. LangGraph waits for both branches
  of a fan-out before running the join node, so the analysis node always sees both the
  cleaned data *and* the plan.
* **The ⇄ loop** — `data_analysis` never writes code. When it has none, or when
  execution raised, it routes back to `llm_integration`. This is the single most
  valuable part of the design: an LLM that gets a real traceback fixes its own
  `KeyError` on the second attempt most of the time.
* **Two edges into the report** — the normal path goes through `interpret`, so the
  report has a narrative. The direct `data_analysis → final_report` edge is the
  give-up path taken only when the retry budget is exhausted *and* nothing usable was
  computed: you still get a report, containing the cleaning log and the last traceback
  instead of findings.

---

## 2. Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Core dependencies: `langgraph`, `langchain-core`, `langchain-openai`,
`langchain-ollama`, `pandas`, `matplotlib`.

---

## 3. Step 1 — Configuration and the dual LLM backend

Every run picks one of two backends. This is the only place in the codebase that knows
the difference; every node is written once against the LangChain interface.

**Option A — local models via Ollama** (free, private, no key):

```powershell
ollama serve            # leave running in its own terminal
ollama pull qwen2.5-coder:7b
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_BASE_URL=http://localhost:11434
```

**Option B — the OpenAI API** (stronger code generation, costs money):

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

The factory in [agent/llm.py](agent/llm.py) resolves this:

```python
if cfg.provider == "ollama":
    llm = ChatOllama(model=..., base_url=..., temperature=..., num_ctx=8192)
else:
    llm = ChatOpenAI(model=..., api_key=..., temperature=..., timeout=...)
```

Two details that matter in practice:

* **JSON mode** is requested natively on both sides (`format="json"` for Ollama,
  `response_format` for OpenAI) for the planning call. It is still backed by a
  forgiving parser, because a 7B model will occasionally ignore it.
* **Forgiving extraction.** `extract_code()` and `extract_json()` cope with fenced
  blocks, unfenced replies and prose wrappers. Skipping this is the number one reason
  local models "don't work" in agent tutorials.

Check the backend before a real run:

```powershell
python -m agent --check
```

---

## 4. Step 2 — State design

LangGraph state is a `TypedDict` of channels. The one subtlety is the **reducers**:

```python
log: Annotated[list[str], operator.add]
artifacts: Annotated[list[str], operator.add]
```

`operator.add` makes those channels *append* rather than overwrite. Without it, the
two branches of the fan-out would write `log` in the same superstep and LangGraph
would raise `InvalidUpdateError`. Everything else is last-write-wins, which is what
you want for `code`, `results` and `stage`.

Two channels drive the routing:

* `stage` — which of the four jobs `llm_integration` should perform. It is set by
  whoever routes *into* that node.
* `next_hop` — where `data_analysis` decided to go. The router function just reads it,
  so the decision logic lives in the node and stays testable.

See [agent/state.py](agent/state.py).

---

## 5. Step 3 — Cleaning (deterministic on purpose)

[agent/profiling.py](agent/profiling.py) does six conservative passes: snake_case
column names, drop all-null rows/columns, de-duplicate, trim whitespace, recover
numeric and datetime types from strings (`"$5,794.43"` becomes `5794.43`), and
**report** missing values rather than imputing them.

It deliberately does not call an LLM. Cleaning must be reproducible and must be
described precisely in the report, so a model choosing silently to drop rows would be
a liability. What the model gets instead is `profile_dataframe()` — a compact schema
with dtypes, null counts, ranges and five example rows. That profile is what makes
generated code reference *real* column names.

> **Note on the profile:** it costs a few hundred tokens and saves most of the
> `KeyError` retries. Never hand a model raw `df.head().to_string()` and hope.

---

## 6. Step 4 — The LLM Integration node

One node, four stages, dispatched on `state["stage"]`
([agent/nodes/llm_integration.py](agent/nodes/llm_integration.py)):

| Stage | Entered from | Produces |
|---|---|---|
| `plan` | Input (in parallel with cleaning) | 3–5 concrete steps, as JSON |
| `codegen` | Data Analysis, when there is nothing to run | a Python block |
| `repair` | Data Analysis, after a traceback | a corrected Python block |
| `interpret` | Data Analysis, after success | the Markdown findings section |

The prompts ([agent/prompts.py](agent/prompts.py)) are written for small models: short
system messages, one job per call, an explicit output contract, no chain-of-thought.
The codegen prompt states the execution contract exactly — `df`, `pd`, `np`, `plt`,
`RESULTS` and `save_fig()` are pre-supplied; imports, file I/O and `plt.show()` are
forbidden.

Every generated block is written to `runs/<id>/generated_attempt_N.py` before it runs,
so a failed run is fully inspectable afterwards.

---

## 7. Step 5 — Executing generated code safely

Never `exec()` model output in the agent process. [agent/sandbox.py](agent/sandbox.py)
instead:

1. **Screens** the code against a deny-list (`os`, `subprocess`, `requests`, `eval`,
   `__import__`, file writes, `to_csv`). A violation is returned as an error string,
   so it feeds the repair loop like any other failure.
2. **Wraps** it in a fixed harness that pre-imports the environment, wraps the
   generated body in `try/except`, and serialises `RESULTS` — DataFrames and Series
   included — to `result.json`. Crucially it serialises **after** the `except`, so a
   crash at line 60 still preserves everything computed in lines 1–59.
   The harness also has an **auto-capture** step: if `RESULTS` is empty it adopts any
   DataFrame, Series or scalar left in the namespace. Small models very often compute
   the right aggregate and then only `print()` it — this recovers that work instead of
   burning a retry on it.
3. **Runs** it via `subprocess.run([sys.executable, script], timeout=...)`. A fresh
   interpreter means a crash, a hang or a `sys.exit` costs one subprocess, not the
   graph. The timeout catches accidental quadratic loops.

The harness returns the traceback verbatim. That string is exactly what the repair
prompt receives — the quality of the loop depends on not summarising it.

> **Scope of the sandbox.** This is a guard against *accidents* — a hallucinated
> `import os`, a runaway loop. It is not a security boundary against deliberately
> hostile code. For untrusted input, run the subprocess in a container or a
> `firejail`/`bwrap` jail with no network.

---

## 8. Step 6 — The analysis node and the routing

`data_analysis` is the loop driver ([agent/nodes/analysis.py](agent/nodes/analysis.py)):

```python
if not state.get("pending_code"):        # nothing to run yet
    return {"stage": "codegen",  "next_hop": "llm_integration", ...}

result = run_analysis(...)               # execute in the sandbox

if result.results:                       # keep partial work from a crash
    update["results"] = result.results

if result.ok:                            # -> narrate it
    return {"stage": "interpret", "next_hop": "llm_integration", ...}
if attempt < settings.max_analysis_retries:
    return {"stage": "repair",    "next_hop": "llm_integration", ...}
if result.results:                       # budget spent but something survived
    return {"stage": "interpret", "next_hop": "llm_integration", ...}
return {"next_hop": "final_report", ...}  # nothing to show; report the failure
```

Note the third branch. Exhausting the retry budget is not automatically a failed run:
if partial results survived, they are still worth narrating. Only a run with *nothing*
to show takes the direct edge to the report.

`MAX_ANALYSIS_RETRIES` is the safety valve. It caps cost, and combined with LangGraph's
`recursion_limit` (40 here) it makes an infinite generate-fail-generate cycle
impossible.

---

## 9. Step 7 — Compiling the graph

```python
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "data_cleaning")       # fan-out
builder.add_edge("ingest", "llm_integration")     # fan-out
builder.add_edge("data_cleaning", "data_analysis")            # join
builder.add_conditional_edges("llm_integration", route_after_llm,
                              {"data_analysis": ..., "final_report": ...})
builder.add_conditional_edges("data_analysis", route_after_analysis,
                              {"llm_integration": ..., "final_report": ...})
builder.add_edge("final_report", END)
```

`run_agent()` streams with `stream_mode=["updates", "values"]` — `updates` drives the
live CLI progress lines, `values` yields the merged state so the caller gets the final
result without a second `invoke()`.

To make runs resumable, pass a checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver
graph = build_graph(checkpointer=MemorySaver())
```

---

## 10. Step 8 — Testing without a model

[tests/test_graph_offline.py](tests/test_graph_offline.py) swaps in a scripted
`FakeLLM` that returns **deliberately broken code on the first attempt** and correct
code on the repair. It then asserts that the fan-out ran, the repair loop fired,
exactly two attempts were used, a figure was saved and the report reached disk.

This is the test worth having: it exercises the wiring, the reducers, the sandbox and
the loop in about two seconds, with no model, no key and no network.

```powershell
.venv\Scripts\python.exe tests\test_graph_offline.py
```

---

## 11. Running it

```powershell
# local model
python -m agent -d data\sample_sales.csv -p "Which region generates the most revenue?" --provider ollama

# OpenAI
python -m agent -d data\sample_sales.csv -p "Which region generates the most revenue?" --provider openai --model gpt-4o-mini
```

Output lands in `runs/<timestamp>_<id>/`: `report.md`, `figures/`, `clean.csv`,
`clean_profile.json`, and every `generated_attempt_N.py` / `analysis_attempt_N.py`.

---

## 12. Choosing a local model

Code generation is the hard part of this workflow — a chat-tuned 7B model that writes
prose instead of pandas will burn every retry.

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:7b` | ~4.7 GB | Best default. Code-tuned, follows the output contract. |
| `qwen2.5-coder:14b` | ~9 GB | Noticeably fewer repair loops, if you have the VRAM. |
| `deepseek-coder-v2:16b` | ~9 GB | Strong pandas; MoE, so fast for its size. |
| `llama3.1:8b` | ~4.7 GB | Good narration, weaker codegen — expect more retries. |

Keep `LLM_TEMPERATURE` at `0.1`. Code generation is not a creative task, and higher
temperatures mainly produce invented column names.

**Observed behaviour on a 3B model.** `llama3.2:3b` was used to validate the loop
end-to-end. It failed attempts 1 and 2 (plotting a DataFrame that still held a string
column, then passing `autopct` to a bar chart) and succeeded on attempt 3, producing
correct revenue totals and two charts. That is the loop doing its job — but it also
shows why a code-tuned 7B model is the better default: it usually lands on attempt 1.

---

## 13. Where to take it next

* **More tools.** Add a `statistical_tests` node, or give `llm_integration` a real
  tool-calling loop instead of a single codegen call.
* **Human in the loop.** Compile with a checkpointer and
  `interrupt_before=["data_analysis"]` to review generated code before it executes.
* **Streamlit front end.** `run_agent()` already accepts an `on_event` callback, so
  progress streams into a UI with no changes to the graph.
* **Multi-dataset joins.** Extend `ingest` to accept several paths and pass a dict of
  DataFrames into the sandbox harness.
