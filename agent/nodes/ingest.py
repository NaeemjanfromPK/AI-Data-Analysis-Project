"""Node 1 - "Input Data + User Prompt".

Loads the dataset, profiles it as-is, and prepares the run directory that every
later node writes artifacts into.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..profiling import load_dataframe, profile_dataframe
from ..state import AgentState


def ingest_node(state: AgentState) -> dict:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(state["data_path"])
    profile = profile_dataframe(df)

    (run_dir / "raw_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return {
        "raw_profile": profile,
        "log": [
            f"[ingest] Loaded {state['data_path']} "
            f"({profile['rows']} rows x {profile['columns']} columns)."
        ],
        "artifacts": [str(run_dir / "raw_profile.json")],
    }
