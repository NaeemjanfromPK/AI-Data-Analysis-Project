"""Dataset loading, profiling and deterministic cleaning.

Kept free of LLM calls on purpose: the cleaning branch of the workflow must be
reproducible and cheap, so the model only ever *reads* the profile it produces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOADERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".txt": lambda p: pd.read_csv(p, sep=None, engine="python"),
    ".json": lambda p: pd.read_json(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
}


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type {path.suffix!r}. Supported: {', '.join(sorted(_LOADERS))}"
        )
    return loader(path)


def _py(value: Any) -> Any:
    """Make numpy/pandas scalars JSON-serialisable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def profile_dataframe(df: pd.DataFrame, *, sample_rows: int = 5) -> dict[str, Any]:
    """A compact, token-cheap description of a dataframe for the LLM."""
    columns = []
    for name in df.columns:
        col = df[name]
        info: dict[str, Any] = {
            "name": str(name),
            "dtype": str(col.dtype),
            "nulls": int(col.isna().sum()),
            "unique": int(col.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(col) and col.notna().any():
            info["min"] = _py(col.min())
            info["max"] = _py(col.max())
            info["mean"] = _py(col.mean())
        elif col.notna().any():
            info["examples"] = [str(v) for v in col.dropna().unique()[:5]]
        columns.append(info)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "schema": columns,
        "head": df.head(sample_rows).astype(str).to_dict(orient="records"),
    }


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Conservative, explainable cleaning. Returns (clean_df, change_log)."""
    log: list[str] = []
    out = df.copy()

    # 1. Normalise column names -> snake_case, unique.
    renames: dict[str, str] = {}
    seen: set[str] = set()
    for col in out.columns:
        new = (
            str(col).strip().lower()
            .replace("%", "pct").replace("#", "num")
            .replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")
        )
        new = "".join(ch for ch in new if ch.isalnum() or ch == "_").strip("_") or "col"
        base, i = new, 1
        while new in seen:
            i += 1
            new = f"{base}_{i}"
        seen.add(new)
        if new != col:
            renames[col] = new
    if renames:
        out = out.rename(columns=renames)
        log.append(f"Normalised {len(renames)} column name(s) to snake_case.")

    # 2. Drop fully-empty rows and columns.
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        log.append(f"Dropped {len(empty_cols)} all-null column(s): {', '.join(empty_cols)}.")
    before = len(out)
    out = out.dropna(how="all")
    if len(out) != before:
        log.append(f"Dropped {before - len(out)} all-null row(s).")

    # 3. De-duplicate.
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"Removed {before - len(out)} exact duplicate row(s).")

    # 4. Trim whitespace on text columns.
    # pandas >= 3.0 gives text columns the `str` dtype; <3.0 uses `object`.
    text_cols = [
        c for c in out.columns
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c])
    ]
    for col in text_cols:
        out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    if text_cols:
        log.append(f"Trimmed whitespace in {len(text_cols)} text column(s).")

    # 5. Opportunistic type recovery: numeric-looking and date-looking strings.
    for col in text_cols:
        series = out[col].dropna()
        if series.empty:
            continue
        numeric = pd.to_numeric(
            out[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
            errors="coerce",
        )
        if numeric.notna().sum() >= 0.9 * len(series):
            out[col] = numeric
            log.append(f"Converted '{col}' to numeric.")
            continue
        if any(tok in col for tok in ("date", "time", "day", "month", "year")):
            parsed = pd.to_datetime(out[col], errors="coerce", format="mixed")
            if parsed.notna().sum() >= 0.9 * len(series):
                out[col] = parsed
                log.append(f"Parsed '{col}' as datetime.")

    # 6. Report (do not impute) remaining missing values.
    missing = {c: int(n) for c, n in out.isna().sum().items() if n}
    if missing:
        summary = ", ".join(f"{c}={n}" for c, n in sorted(missing.items(), key=lambda kv: -kv[1])[:8])
        log.append(f"Missing values left in place for the analysis step: {summary}.")

    out = out.reset_index(drop=True)
    if not log:
        log.append("Dataset already clean; no changes applied.")
    return out, log
