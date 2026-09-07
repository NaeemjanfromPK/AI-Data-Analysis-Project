"""Node 3 - "LLM Integration".

One node, four jobs, selected by ``state["stage"]``:

    plan       reached from Input, in parallel with Data Cleaning
    codegen    reached from Data Analysis when there is nothing to run yet
    repair     reached from Data Analysis when the last run raised
    interpret  reached from Data Analysis after a successful run

Backend (local Ollama vs. OpenAI API) is resolved by :mod:`agent.llm`; nothing
in this module knows which one is in use.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..verify import build_verified_facts
from .. import prompts
from ..llm import build_llm, chat, extract_code, extract_json
from ..state import AgentState


def _schema_text(profile: dict) -> str:
    lines = []
    for col in profile.get("schema", []):
        bits = [f"- {col['name']} ({col['dtype']})"]
        if col.get("nulls"):
            bits.append(f"nulls={col['nulls']}")
        if "min" in col:
            bits.append(f"range=[{col['min']}, {col['max']}]")
        elif col.get("examples"):
            bits.append("examples=" + ", ".join(col["examples"][:3]))
        lines.append(" | ".join(bits))
    return "\n".join(lines) or "(no columns)"


def _truncate(text: str, limit: int = 6000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


# ---------------------------------------------------------------- stages ---
def _plan(state: AgentState) -> dict:
    profile = state.get("raw_profile", {})
    compact = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "schema": _schema_text(profile),
        "head": profile.get("head", [])[:3],
    }
    reply = chat(
        prompts.PLANNER_SYSTEM,
        prompts.PLANNER_USER.format(
            user_prompt=state["user_prompt"], profile=json.dumps(compact, indent=2)
        ),
        llm=build_llm(json_mode=True),
    )
    parsed = extract_json(reply)
    steps = [str(s) for s in parsed.get("plan", []) if str(s).strip()]
    if not steps:  # small models occasionally return prose; degrade gracefully
        steps = [ln.lstrip("-*0123456789. ").strip()
                 for ln in reply.splitlines() if len(ln.strip()) > 12][:6]
    if not steps:
        steps = ["Summarise the dataset and answer the question with descriptive statistics."]

    return {"plan": steps, "log": [f"[llm:plan] {len(steps)} step(s) planned."]}


def _codegen(state: AgentState) -> dict:
    profile = state.get("clean_profile", {})
    reply = chat(
        prompts.CODEGEN_SYSTEM,
        prompts.CODEGEN_USER.format(
            user_prompt=state["user_prompt"],
            plan="\n".join(f"{i}. {s}" for i, s in enumerate(state.get("plan", []), 1)),
            rows=profile.get("rows", "?"),
            columns=profile.get("columns", "?"),
            schema=_schema_text(profile),
            cleaning_report="\n".join(f"- {c}" for c in state.get("cleaning_report", [])) or "- none",
        ),
        llm=build_llm(),
    )
    code = extract_code(reply)
    return {
        "code": code,
        "pending_code": True,
        "log": [f"[llm:codegen] Generated {len(code.splitlines())} lines of analysis code."],
    }


def _repair(state: AgentState) -> dict:
    reply = chat(
        prompts.REPAIR_SYSTEM,
        prompts.REPAIR_USER.format(
            user_prompt=state["user_prompt"],
            schema=_schema_text(state.get("clean_profile", {})),
            attempt=state.get("attempts", 1),
            code=state.get("code", ""),
            error=_truncate(state.get("exec_error", ""), 3000),
        ),
        llm=build_llm(),
    )
    code = extract_code(reply)
    return {
        "code": code,
        "pending_code": True,
        "log": [f"[llm:repair] Rewrote code after attempt {state.get('attempts', 1)}."],
    }


def _interpret(state: AgentState) -> dict:
    verified_facts = build_verified_facts(state.get("results", {}))
    Path(state["run_dir"]).joinpath("verified_facts_debug.txt").write_text(verified_facts, encoding="utf-8")
    text = chat(
        prompts.INTERPRET_SYSTEM,
        prompts.INTERPRET_USER.format(
            user_prompt=state["user_prompt"],
            plan="\n".join(f"{i}. {s}" for i, s in enumerate(state.get("plan", []), 1)),
            results=_truncate(json.dumps(state.get("results", {}), indent=2), 8000),
            verified_facts=verified_facts,
            stdout=_truncate(state.get("exec_stdout", ""), 3000),
            cleaning_report="\n".join(f"- {c}" for c in state.get("cleaning_report", [])) or "- none",
        ),
        llm=build_llm(),
    )
    return {"interpretation": text, "log": ["[llm:interpret] Findings drafted."]}


_STAGES = {"plan": _plan, "codegen": _codegen, "repair": _repair, "interpret": _interpret}


def llm_integration_node(state: AgentState) -> dict:
    stage = state.get("stage", "plan")
    handler = _STAGES.get(stage, _plan)
    update = handler(state)

    # Persist every generated artefact so a run is fully auditable afterwards.
    run_dir = Path(state["run_dir"])
    if "code" in update:
        path = run_dir / f"generated_attempt_{state.get('attempts', 0) + 1}.py"
        path.write_text(update["code"], encoding="utf-8")
        update.setdefault("artifacts", []).append(str(path))

    update["stage"] = stage  # routers downstream read the stage that just ran
    return update


def route_after_llm(state: AgentState) -> str:
    """LLM Integration -> Final Report (after interpreting) or -> Data Analysis."""
    return "final_report" if state.get("stage") == "interpret" else "data_analysis"
