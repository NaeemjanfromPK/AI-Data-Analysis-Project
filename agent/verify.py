"""Compute verified, ground-truth facts from analysis RESULTS before the
LLM writes its findings -- so it states pre-computed numbers instead of
eyeballing a table and getting the arithmetic wrong.

Works on the same JSON-ified shape sandbox.py already produces:
    {"__table__": [ {col: val, ...}, ... ], "__columns__": [...], "__shape__": [...]}
    {"__series__": {label: value, ...}}
    or a plain scalar (int/float/str/bool)

Only acts on tables/series that have exactly one clearly-numeric value
column -- ambiguous shapes are skipped rather than guessed at.
"""
from __future__ import annotations


def _numeric_column(rows: list[dict], columns: list[str]) -> str | None:
    """Return the single column that looks purely numeric, else None."""
    numeric_cols = []
    for col in columns:
        values = [r.get(col) for r in rows if r.get(col) is not None]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            numeric_cols.append(col)
    return numeric_cols[0] if len(numeric_cols) == 1 else None


def _label_columns(columns: list[str], numeric_col: str) -> list[str]:
    return [c for c in columns if c != numeric_col]


def verify_table(name: str, table_value: dict) -> str | None:
    """Produce a verified-facts sentence for one __table__ result, or None."""
    rows = table_value.get("__table__", [])
    columns = table_value.get("__columns__", [])
    if not rows or not columns:
        return None

    numeric_col = _numeric_column(rows, columns)
    if not numeric_col:
        return None
    label_cols = _label_columns(columns, numeric_col)
    if not label_cols:
        return None

    def label_of(row: dict) -> str:
        return " / ".join(str(row.get(c, "")) for c in label_cols)

    ranked = sorted(rows, key=lambda r: r.get(numeric_col, 0), reverse=True)
    top, bottom = ranked[0], ranked[-1]
    total = sum(r.get(numeric_col, 0) for r in rows)

    lines = [
        f"VERIFIED (do not recompute, use these exact figures) for '{name}' by {numeric_col}:",
        f"  - Highest: {label_of(top)} = {top.get(numeric_col)}",
        f"  - Lowest: {label_of(bottom)} = {bottom.get(numeric_col)}",
        f"  - Total across all {len(rows)} rows: {round(total, 2)}",
    ]
    if len(label_cols) > 1:
        # Also roll up by the first label column alone (e.g. region, ignoring category)
        rollup: dict[str, float] = {}
        for r in rows:
            key = str(r.get(label_cols[0], ""))
            rollup[key] = rollup.get(key, 0) + (r.get(numeric_col) or 0)
        ranked_rollup = sorted(rollup.items(), key=lambda kv: kv[1], reverse=True)
        top_group, top_val = ranked_rollup[0]
        lines.append(
            f"  - Grouped by {label_cols[0]} only: {top_group} leads with {round(top_val, 2)} "
            f"(full ranking: {', '.join(f'{k}={round(v, 2)}' for k, v in ranked_rollup)})"
        )
    return "\n".join(lines)


def verify_series(name: str, series_value: dict) -> str | None:
    series = series_value.get("__series__", {})
    numeric_items = {k: v for k, v in series.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not numeric_items:
        return None
    ranked = sorted(numeric_items.items(), key=lambda kv: kv[1], reverse=True)
    top_k, top_v = ranked[0]
    bottom_k, bottom_v = ranked[-1]
    return (
        f"VERIFIED (do not recompute, use these exact figures) for '{name}':\n"
        f"  - Highest: {top_k} = {top_v}\n"
        f"  - Lowest: {bottom_k} = {bottom_v}\n"
        f"  - Full ranking: {', '.join(f'{k}={v}' for k, v in ranked)}"
    )


def build_verified_facts(results: dict) -> str:
    """Return a text block of verified facts for every table/series in results.

    Pass this straight into the INTERPRET prompt so the LLM quotes it
    instead of re-deriving numbers from the raw table by eye.
    """
    blocks = []
    for name, value in results.items():
        if isinstance(value, dict) and "__table__" in value:
            block = verify_table(name, value)
        elif isinstance(value, dict) and "__series__" in value:
            block = verify_series(name, value)
        else:
            continue
        if block:
            blocks.append(block)

    if not blocks:
        return "No table/series results available for automatic fact verification."
    return "\n\n".join(blocks)