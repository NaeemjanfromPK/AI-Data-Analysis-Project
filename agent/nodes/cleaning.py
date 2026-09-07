"""Node 2 - "Data Cleaning" (top branch of the fan-out).

Deterministic, LLM-free. Runs in parallel with the planning pass of the LLM
Integration node and joins at Data Analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..profiling import clean_dataframe, load_dataframe, profile_dataframe
from ..state import AgentState


def data_cleaning_node(state: AgentState) -> dict:
    run_dir = Path(state["run_dir"])
    df = load_dataframe(state["data_path"])
    clean, report = clean_dataframe(df)

    pkl_path = run_dir / "clean.pkl"   # dtype-preserving, consumed by the sandbox
    csv_path = run_dir / "clean.csv"   # human-inspectable copy
    clean.to_pickle(pkl_path)
    clean.to_csv(csv_path, index=False)

    profile = profile_dataframe(clean)
    (run_dir / "clean_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return {
        "clean_path": str(pkl_path),
        "clean_profile": profile,
        "cleaning_report": report,
        "log": [f"[cleaning] {len(report)} action(s); {profile['rows']} rows survived."],
        "artifacts": [str(pkl_path), str(csv_path), str(run_dir / "clean_profile.json")],
    }
