"""
Streamlit UI for the AI Data Analyst agent.

Save this file in the project ROOT (same folder as the `agent/` package),
then run:
    streamlit run streamlit_app.py

Make sure your `python_basic` conda env is active and Ollama is running
before launching.
"""
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from agent.config import settings
from agent.graph import run_agent


# ---------------------------------------------------------------- backend check
def _ollama_reachable() -> bool:
    """Quick probe to see if Ollama is actually running."""
    import socket
    try:
        host = settings.ollama_base_url.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(settings.ollama_base_url.split(":")[-1]) if ":" in settings.ollama_base_url else 11434
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


_backend_ok = True
if settings.provider == "ollama" and not _ollama_reachable():
    _backend_ok = False
    st.warning(
        "⚠️ **Ollama is not running.** "
        "This app requires a local Ollama server. "
        "Please install Ollama and run `ollama serve` in a terminal, then refresh this page.  \n\n"
        "📖 [Setup instructions](https://github.com/NaeemjanfromPK/AI-Data-Analysis-Project#backend-a--local-via-ollama)"
    )




st.set_page_config(page_title="AI Data Analyst", layout="wide")

DEFAULT_PROMPT = (
    "Give me a general overview of this dataset: the key trends, the top and "
    "bottom performers across the main categories, and anything unusual worth "
    "flagging to a stakeholder."
)

SAMPLE_DATASET = Path("data/sample_sales.csv")

# ---------------------------------------------------------------- helpers
def parse_report_sections(report_text: str) -> dict:
    """Split the markdown report into level-2 ("## ") sections keyed by title."""
    sections = {}
    parts = re.split(r"(?m)^## (.+)$", report_text)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections[title] = body.strip()
    return sections


def parse_subsections(body: str) -> dict:
    """Split a section's body into level-3 ("### ") subsections keyed by title."""
    sub = {}
    parts = re.split(r"(?m)^### (.+)$", body)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sub[title] = content.strip()
    return sub


# def strip_image_markdown(text: str) -> str:
#     """Remove ![...](...) image tags -- figures are rendered separately via st.image,
#     since a filesystem-relative path in markdown does not resolve in the browser."""
#     return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).strip()
def strip_image_markdown(text: str) -> str:
    """Remove image tags -- figures are rendered separately via st.image,
    since a filesystem-relative path in markdown does not resolve in the browser."""
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).strip()




# ---------------------------------------------------------------- main area
st.title("📊 AI Data Analyst")
st.caption("Local agent powered by Ollama — upload a file, ask a question, get a report.")

st.subheader("1. Choose your data")
data_source = st.radio(
    "Data source",
    ["Upload my own file", "Use sample data"],
    horizontal=True,
    label_visibility="collapsed",
)

data_path: Path | None = None

if data_source == "Upload my own file":
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded_file is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "ai_agent_uploads"
        tmp_dir.mkdir(exist_ok=True)
        data_path = tmp_dir / uploaded_file.name
        data_path.write_bytes(uploaded_file.getbuffer())
else:
    if SAMPLE_DATASET.exists():
        st.info(f"Using bundled sample dataset: `{SAMPLE_DATASET}`")
        data_path = SAMPLE_DATASET
    else:
        st.error(f"Sample dataset not found at `{SAMPLE_DATASET}`. Upload a file instead.")

st.subheader("2. Ask a question (optional)")
user_prompt = st.text_area(
    "What do you want to know about this data?",
    placeholder=f"e.g. Which region generates the most revenue, and how do ratings compare across categories?\n\nLeave blank for: \"{DEFAULT_PROMPT}\"",
    label_visibility="collapsed",
)

# run_button = st.button("Run analysis", type="primary", disabled=data_path is None)
run_button = st.button("Run analysis", type="primary", disabled=(data_path is None or not _backend_ok))

# ---------------------------------------------------------------- run agent
# Results are stashed in session_state so they SURVIVE later reruns caused
# by clicking Export / Show more -- those are normal Streamlit buttons and
# every button click reruns the whole script from top to bottom.
if run_button:
    effective_prompt = user_prompt.strip() or DEFAULT_PROMPT
    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    progress_box = st.empty()
    log_lines: list[str] = []

    def on_event(node_name, update):
        log_lines.append(f"✔ {node_name}")
        progress_box.code("\n".join(log_lines))

    with st.spinner("Running the agent... this can take a minute or two on a local model."):
        try:
            run_agent(str(data_path), effective_prompt, run_id=run_id, on_event=on_event)
        except Exception as e:
            st.error(f"The agent raised an error: {e}")
            st.stop()

    run_dir = settings.runs_dir / run_id
    report_path = run_dir / "report.md"

    if not report_path.exists():
        st.error("No report was generated. Check the terminal running `streamlit run` for the underlying error.")
        st.stop()

    # Save everything needed to redraw this result on future reruns.
    st.session_state["last_run"] = {
        "run_id": run_id,
        "data_path": str(data_path),
        "run_dir": str(run_dir),
        "report_text": report_path.read_text(encoding="utf-8"),
    }
    # Reset the "show more" toggle for a fresh run.
    st.session_state.pop(f"show_backend_details_{run_id}", None)

# ---------------------------------------------------------------- render last result (if any)
last_run = st.session_state.get("last_run")

if last_run:
    run_id = last_run["run_id"]
    run_dir = Path(last_run["run_dir"])
    report_text = last_run["report_text"]
    figures_dir = run_dir / "figures"
    clean_csv_path = run_dir / "clean.csv"

    st.success("Analysis complete!")

    sections = parse_report_sections(report_text)

    # Section numbers shift run-to-run depending on how many result tables
    # were produced, so match section titles by keyword, not a fixed number.
    def _find_section(sections: dict, keyword: str) -> str:
        for title, body in sections.items():
            if keyword.lower() in title.lower():
                return body
        return ""

    def _find_data_cleaning(sections: dict) -> str:
        return _find_section(sections, "Data cleaning")

    findings_body = _find_section(sections, "Findings")
    findings_sub = parse_subsections(findings_body)

    # ---- report header + export buttons -----------------------------
    st.markdown("---")
    st.subheader("Data Analysis Report")
    meta_col, export_col1, export_col2 = st.columns([3, 1, 1])
    with meta_col:
        st.write(f"**Dataset:** `{last_run['data_path']}`")
        st.write(f"**Run ID:** `{run_id}`")
        st.write(
            f"**LLM backend:** `{getattr(settings, 'llm_provider', 'ollama')} / "
            f"{getattr(settings, 'ollama_model', 'qwen2.5-coder:7b')}`"
        )
    with export_col1:
        if clean_csv_path.exists():
            st.download_button(
                "📥 Export clean data",
                data=clean_csv_path.read_bytes(),
                file_name=f"clean_{run_id}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_clean_{run_id}",
            )
    with export_col2:
        st.download_button(
            "📥 Export full report",
            data=report_text.encode("utf-8"),
            file_name=f"report_{run_id}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_report_{run_id}",
        )

    # ---- 1. Charts ----------------------------------------------------
    if figures_dir.exists():
        figs = sorted(figures_dir.glob("*.png"))
        if figs:
            st.subheader("Charts")
            cols = st.columns(2)
            for i, fig_path in enumerate(figs):
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_path.stem)

    # ---- 2. Findings ----------------------------------------------------
    st.subheader("Findings")
    for label in ["Executive summary", "Key findings", "Business impact", "Recommendations"]:
        if label in findings_sub:
            st.markdown(f"#### {label}")
            st.markdown(strip_image_markdown(findings_sub[label]))

    # ---- 3. Backend work (collapsed by default) ------------------------
    st.markdown("---")
    with st.expander("Data Analysis Backend work"):
        cleaning_text = _find_data_cleaning(sections)
        if cleaning_text:
            st.markdown("#### Data cleaning")
            st.markdown(strip_image_markdown(cleaning_text))

        show_more_key = f"show_backend_details_{run_id}"
        if st.button("Show more", key=f"btn_{show_more_key}"):
            st.session_state[show_more_key] = True

        if st.session_state.get(show_more_key):
            plan_text = _find_section(sections, "Analysis plan")
            if plan_text:
                st.markdown("#### Analysis plan")
                st.markdown(strip_image_markdown(plan_text))
            results_text = _find_section(sections, "Results")
            if results_text:
                st.markdown("#### Results")
                st.markdown(strip_image_markdown(results_text))
            if "Caveats" in findings_sub:
                st.markdown("#### Caveats")
                st.markdown(strip_image_markdown(findings_sub["Caveats"]))

# ---------------------------------------------------------------- sidebar
# Targets ONLY the sidebar's content area (below the collapse arrow), so the
# collapse button itself is left alone. Streamlit wraps every element in its
# own container div -- the spacer must grow via THAT wrapper (matched below
# with :has()), not the inner div itself, or flex:1 has no effect.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { height: 100vh; }
    [data-testid="stSidebarUserContent"] { height: 100%; }
    [data-testid="stSidebarUserContent"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    div[data-testid="stElementContainer"]:has(.sidebar-spacer) {
        flex: 1 1 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)
    st.markdown("### Backend")
    st.write(f"**Provider:** `{getattr(settings, 'llm_provider', 'ollama')}`")
    st.write(f"**Model:** `{getattr(settings, 'ollama_model', 'qwen2.5-coder:7b')}`")
    st.info("Runs entirely on your machine. No data leaves your computer.")