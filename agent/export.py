"""Export the agent's RESULTS into a structured .xlsx workbook.

Reads the same result.json that sandbox.py already writes for every run,
so this has no dependency on internal state shapes -- just the run_dir.
One sheet per result (table/series/scalar), plus a Summary sheet listing
every finding at a glance. Designed to be opened directly, or connected
live, in Power BI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel sheet names: <=31 chars, no []:*?/\\, must be unique."""
    cleaned = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "Sheet"
    candidate = cleaned
    i = 1
    while candidate in used:
        suffix = f"_{i}"
        candidate = cleaned[: 31 - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate


def export_results_to_excel(run_dir: str | Path, filename: str = "analysis_export.xlsx") -> Path | None:
    """Read result.json from run_dir and write a structured workbook next to it.

    Returns the output path, or None if there was nothing to export.
    """
    run_dir = Path(run_dir)
    result_file = run_dir / "result.json"
    if not result_file.exists():
        return None

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not results:
        return None

    wb = Workbook()
    summary_ws = wb.active
    summary_ws.title = "Summary"
    summary_ws.append(["Finding", "Type", "Detail"])
    for cell in summary_ws[1]:
        cell.font = Font(bold=True)

    used_names: set[str] = {"Summary"}

    for key, value in results.items():
        detail = ""

        if isinstance(value, dict) and "__table__" in value:
            rows = value["__table__"]
            columns = value.get("__columns__", list(rows[0].keys()) if rows else [])
            sheet_name = _safe_sheet_name(key, used_names)
            ws = wb.create_sheet(sheet_name)
            ws.append(columns)
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for row in rows:
                ws.append([row.get(c, "") for c in columns])
            detail = f"{len(rows)} row(s) x {len(columns)} column(s) -> sheet '{sheet_name}'"

        elif isinstance(value, dict) and "__series__" in value:
            series = value["__series__"]
            sheet_name = _safe_sheet_name(key, used_names)
            ws = wb.create_sheet(sheet_name)
            ws.append(["Label", "Value"])
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for label, val in series.items():
                ws.append([label, val])
            detail = f"{len(series)} item(s) -> sheet '{sheet_name}'"

        else:
            # Scalar (number, string, bool)
            detail = str(value)

        summary_ws.append([key, type(value).__name__ if not isinstance(value, dict) else "table/series", detail])

    for col in summary_ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
        summary_ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)

    out_path = run_dir / filename
    wb.save(out_path)
    return out_path