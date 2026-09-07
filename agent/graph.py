"""Wiring of the LangGraph workflow.

    ┌──────────────────┐        ┌───────────────┐
    │ Input Data +     │───────▶│ Data Cleaning │──────┐
    │ User Prompt      │        └───────────────┘      ▼
    └──────────────────┘                        ┌──────────────────┐
             │                                  │  Data Analysis   │────▶ Final
             │        ┌──────────────────┐      │  using Python    │      Report
             └───────▶│ LLM Integration  │─────▶│                  │        ▲
                      └──────────────────┘◀─────└──────────────────┘        │
                               └───────────────────────────────────────────-┘
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .config import settings
from .nodes import (
    data_analysis_node,
    data_cleaning_node,
    excel_export_node,
    final_report_node,
    ingest_node,
    llm_integration_node,
    route_after_analysis,
    route_after_llm,
)

from .state import AgentState, new_state


def build_graph(checkpointer=None):
    """Compile the workflow. Pass a LangGraph checkpointer to make runs resumable."""
    builder = StateGraph(AgentState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("data_cleaning", data_cleaning_node)
    builder.add_node("llm_integration", llm_integration_node)
    builder.add_node("data_analysis", data_analysis_node)
    builder.add_node("final_report", final_report_node)
    builder.add_node("excel_export", excel_export_node)

    builder.add_edge(START, "ingest")

    # Fan-out: cleaning and the planning pass run concurrently.
    builder.add_edge("ingest", "data_cleaning")
    builder.add_edge("ingest", "llm_integration")

    # Join: both branches land on the analysis node.
    builder.add_edge("data_cleaning", "data_analysis")
    builder.add_conditional_edges(
        "llm_integration",
        route_after_llm,
        {"data_analysis": "data_analysis", "final_report": "final_report"},
    )

    # The generate -> execute -> repair loop, plus the exit to the report.
    builder.add_conditional_edges(
        "data_analysis",
        route_after_analysis,
        {"llm_integration": "llm_integration", "final_report": "final_report"},
    )

    builder.add_edge("final_report", "excel_export") 
    builder.add_edge("excel_export", END)
    return builder.compile(checkpointer=checkpointer)


def make_run_dir(run_id: str) -> Path:
    path = settings.runs_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_agent(data_path: str, user_prompt: str, *, run_id: str | None = None,
              recursion_limit: int = 40, on_event=None) -> AgentState:
    """Execute the whole workflow once and return the final merged state.

    ``on_event(node_name, update)`` is invoked after every node so a CLI or UI
    can render progress live.
    """
    run_id = run_id or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    run_dir = make_run_dir(run_id)

    graph = build_graph()
    initial = new_state(str(data_path), user_prompt, run_id, str(run_dir))

    final: AgentState = dict(initial)  # type: ignore[assignment]
    for mode, chunk in graph.stream(
        initial,
        {"recursion_limit": recursion_limit},
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node_name, update in (chunk or {}).items():
                if on_event and isinstance(update, dict):
                    on_event(node_name, update)
        elif mode == "values" and isinstance(chunk, dict):
            final = chunk  # type: ignore[assignment]
    return final
