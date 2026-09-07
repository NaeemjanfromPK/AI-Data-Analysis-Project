"""Node 4 - "Data Analysis using Python".

The join point of the fan-out (Data Cleaning + LLM Integration) and the driver
of the repair loop. It never writes code itself: it executes, judges the
outcome, and decides whether to bounce back to LLM Integration or move on to
the Final Report.
"""
from __future__ import annotations

from ..config import settings
from ..sandbox import run_analysis
from ..state import AgentState


def data_analysis_node(state: AgentState) -> dict:
    # Nothing to execute yet -> ask LLM Integration for code.
    if not state.get("pending_code"):
        return {
            "stage": "codegen",
            "next_hop": "llm_integration",
            "log": ["[analysis] No code yet; requesting generation."],
        }

    attempt = state.get("attempts", 0) + 1
    result = run_analysis(
        state["code"],
        run_dir=state["run_dir"],
        clean_path=state["clean_path"],
        timeout=settings.sandbox_timeout,
        attempt=attempt,
    )

    update: dict = {
        "attempts": attempt,
        "pending_code": False,
        "exec_stdout": result.stdout,
        "exec_error": "" if result.ok else result.stderr,
        "artifacts": [p for p in (result.script_path, *result.figures) if p],
    }
    # A crash partway through still leaves valid work behind (the harness
    # serialises RESULTS before re-raising). Keep it -- discarding correct
    # aggregates because a later chart failed is the wrong trade.
    if result.results:
        update["results"] = result.results
    if result.figures:
        update["figures"] = result.figures

    if result.ok:
        update |= {
            "stage": "interpret",
            "next_hop": "llm_integration",
            "log": [
                f"[analysis] Attempt {attempt} succeeded: "
                f"{len(result.results)} result(s), {len(result.figures)} figure(s)."
            ],
        }
    elif attempt < settings.max_analysis_retries:
        update |= {
            "stage": "repair",
            "next_hop": "llm_integration",
            "log": [f"[analysis] Attempt {attempt} failed; sending traceback back to the LLM."],
        }
    elif result.results:
        # Budget spent, but partial output survived -- narrate what we do have.
        update |= {
            "stage": "interpret",
            "next_hop": "llm_integration",
            "log": [
                f"[analysis] Budget spent after {attempt} attempt(s); "
                f"interpreting {len(result.results)} partial result(s)."
            ],
        }
    else:
        update |= {
            "next_hop": "final_report",
            "log": [
                f"[analysis] Giving up after {attempt} attempt(s); "
                "reporting the failure instead."
            ],
        }
    return update


def route_after_analysis(state: AgentState) -> str:
    return state.get("next_hop") or "final_report"
