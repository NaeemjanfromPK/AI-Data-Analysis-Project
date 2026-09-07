"""The shared state object that flows through the LangGraph workflow."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Stage = Literal["plan", "codegen", "repair", "interpret"]


class AgentState(TypedDict, total=False):
    """Channels written by the nodes of the graph.

    Reducer notes
    -------------
    ``log`` and ``artifacts`` use ``operator.add`` so the two parallel branches
    (Data Cleaning and LLM Integration) can both append without clobbering each
    other during the fan-out superstep.
    """

    # ---- Input Data + User Prompt -----------------------------------------
    data_path: str
    user_prompt: str
    run_id: str
    run_dir: str

    # ---- Data Cleaning -----------------------------------------------------
    raw_profile: dict[str, Any]
    clean_path: str
    clean_profile: dict[str, Any]
    cleaning_report: list[str]

    # ---- LLM Integration ---------------------------------------------------
    stage: Stage
    plan: list[str]
    code: str
    pending_code: bool
    interpretation: str
    next_hop: str

    # ---- Data Analysis using Python ---------------------------------------
    exec_stdout: str
    exec_error: str
    results: dict[str, Any]
    figures: list[str]
    attempts: int

    # ---- Final Report ------------------------------------------------------
    report_md: str
    report_path: str

    # ---- Cross-cutting -----------------------------------------------------
    log: Annotated[list[str], operator.add]
    artifacts: Annotated[list[str], operator.add]


def new_state(data_path: str, user_prompt: str, run_id: str, run_dir: str) -> AgentState:
    return AgentState(
        data_path=data_path,
        user_prompt=user_prompt,
        run_id=run_id,
        run_dir=run_dir,
        stage="plan",
        pending_code=False,
        next_hop="",
        attempts=0,
        plan=[],
        cleaning_report=[],
        figures=[],
        results={},
        log=[],
        artifacts=[],
    )
