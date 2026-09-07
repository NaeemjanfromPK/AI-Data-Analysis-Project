"""Node 5 - "Final Report Ready".

Assembles the deliverable from every upstream channel: the cleaning log, the
plan, the computed results, the figures and the LLM's narrative.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..config import settings
from ..state import AgentState


def _render_results(results: dict) -> str:
    if not results:
        return "_No numeric results were produced._"
    lines: list[str] = []
    for key, value in results.items():
        title = str(key).replace("_", " ").strip().capitalize()
        if isinstance(value, dict) and "__table__" in value:
            rows = value["__table__"]
            cols = value.get("__columns__") or (list(rows[0]) if rows else [])
            shape = value.get("__shape__", [len(rows), len(cols)])
            lines.append(f"**{title}** — {shape[0]} rows x {shape[1]} columns (first {len(rows)} shown)\n")
            if rows and cols:
                lines.append("| " + " | ".join(str(c) for c in cols) + " |")
                lines.append("|" + "|".join(["---"] * len(cols)) + "|")
                for row in rows[:15]:
                    lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
            lines.append("")
        elif isinstance(value, dict) and "__series__" in value:
            lines.append(f"**{title}**\n")
            lines.append("| key | value |")
            lines.append("|---|---|")
            for k, v in list(value["__series__"].items())[:15]:
                lines.append(f"| {k} | {v} |")
            lines.append("")
        elif isinstance(value, (dict, list)):
            lines.append(f"**{title}**\n\n```json\n{json.dumps(value, indent=2)[:1500]}\n```\n")
        else:
            lines.append(f"- **{title}:** {value}")
    return "\n".join(lines)


def final_report_node(state: AgentState) -> dict:
    run_dir = Path(state["run_dir"])
    figures = state.get("figures", [])
    failed = bool(state.get("exec_error")) and not state.get("results")

    parts = [
        f"# Data Analysis Report",
        "",
        f"**Question:** {state['user_prompt']}",
        "",
        f"| | |",
        f"|---|---|",
        f"| Dataset | `{state['data_path']}` |",
        f"| Run ID | `{state['run_id']}` |",
        f"| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| LLM backend | `{settings.provider}` / `{settings.model_name()}` |",
        f"| Analysis attempts | {state.get('attempts', 0)} |",
        "",
        "## 1. Data cleaning",
        "",
    ]
    parts += [f"- {item}" for item in state.get("cleaning_report", ["- none"])]

    raw, clean = state.get("raw_profile", {}), state.get("clean_profile", {})
    if raw and clean:
        parts += [
            "",
            f"Shape: {raw.get('rows')} x {raw.get('columns')} → "
            f"**{clean.get('rows')} x {clean.get('columns')}**",
        ]

    parts += ["", "## 2. Analysis plan", ""]
    parts += [f"{i}. {step}" for i, step in enumerate(state.get("plan", []), 1)] or ["_none_"]

    parts += ["", "## 3. Results", "", _render_results(state.get("results", {}))]

    if figures:
        parts += ["", "## 4. Figures", ""]
        for fig in figures:
            name = Path(fig).stem.replace("_", " ")
            rel = Path(fig).relative_to(run_dir) if Path(fig).is_relative_to(run_dir) else Path(fig)
            parts += [f"**{name}**", "", f"![{name}]({rel.as_posix()})", ""]

    parts += ["", f"## {5 if figures else 4}. Findings", ""]
    if failed:
        parts += [
            "> The analysis code could not be executed successfully after "
            f"{state.get('attempts', 0)} attempt(s). Last error:",
            "",
            "```",
            (state.get("exec_error", "") or "")[-2000:],
            "```",
        ]
    else:
        parts += [state.get("interpretation", "_No interpretation produced._")]

    if state.get("exec_stdout"):
        parts += [
            "",
            "<details><summary>Analysis console output</summary>",
            "",
            "```",
            state["exec_stdout"][-3000:],
            "```",
            "",
            "</details>",
        ]

    parts += [
        "",
        "---",
        "",
        "<details><summary>Execution trace</summary>",
        "",
        *[f"- {line}" for line in state.get("log", [])],
        "",
        "</details>",
    ]

    report = "\n".join(parts)
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "report_md": report,
        "report_path": str(report_path),
        "log": [f"[report] Written to {report_path}."],
        "artifacts": [str(report_path)],
    }
