"""End-to-end exercise of the graph with a scripted fake LLM.

Proves the wiring -- fan-out, join, the repair loop and both routes into the
report -- without needing Ollama or an OpenAI key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage

import agent.nodes.llm_integration as li
from agent.config import settings
from agent.graph import run_agent

BAD_CODE = """```python
RESULTS["oops"] = df["column_that_does_not_exist"].sum()
```"""

GOOD_CODE = """```python
rev = df.groupby("region")["revenue_usd"].sum().sort_values(ascending=False)
RESULTS["revenue_by_region"] = rev
RESULTS["total_revenue"] = float(df["revenue_usd"].sum())
RESULTS["avg_rating"] = float(df["customer_rating"].mean())
print("revenue by region:", rev.to_dict())

rev.plot(kind="bar", color="#4C78A8")
plt.title("Revenue by region")
plt.ylabel("Revenue (USD)")
save_fig("revenue_by_region")
```"""


class FakeLLM:
    """Replays a fixed script; records which stages were asked for."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, messages):
        system = messages[0].content
        if "analysis plans" in system:
            self.calls.append("plan")
            return AIMessage(content='{"plan": ["Sum revenue by region", "Plot it"]}')
        if "debugging" in system:
            self.calls.append("repair")
            return AIMessage(content=GOOD_CODE)
        if "code generator" in system:
            self.calls.append("codegen")
            return AIMessage(content=BAD_CODE)   # first attempt fails on purpose
        self.calls.append("interpret")
        return AIMessage(content="### Key findings\n- North leads revenue.")


def main() -> int:
    fake = FakeLLM()
    li.build_llm = lambda *a, **k: fake          # noqa: ARG005 - test double
    settings.max_analysis_retries = 3

    state = run_agent(
        "data/sample_sales.csv",
        "Which region generates the most revenue, and how do ratings compare?",
        run_id="offline_test",
    )

    checks = {
        "cleaning ran": bool(state.get("cleaning_report")),
        "plan produced": len(state.get("plan", [])) >= 2,
        "repair loop fired": "repair" in fake.calls,
        "analysis succeeded": bool(state.get("results")),
        "figure saved": len(state.get("figures", [])) == 1,
        "interpretation written": bool(state.get("interpretation")),
        "report on disk": Path(state.get("report_path", "x")).exists(),
        "two attempts used": state.get("attempts") == 2,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  LLM stages in order: {fake.calls}")
    print(f"  Report: {state.get('report_path')}")

    failed = [n for n, ok in checks.items() if not ok]
    if failed:
        print(f"\nFAILED: {failed}")
        if state.get("exec_error"):
            print(state["exec_error"][-1500:])
        return 1
    print("\nAll offline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
