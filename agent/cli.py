"""Command-line entry point.

    python -m agent --data data/sample_sales.csv --prompt "..." --provider ollama
    python -m agent --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import settings
from .llm import health_check

console = Console()

_NODE_STYLE = {
    "ingest": "cyan",
    "data_cleaning": "green",
    "llm_integration": "magenta",
    "data_analysis": "yellow",
    "final_report": "bold blue",
}


def _on_event(node: str, update: dict) -> None:
    style = _NODE_STYLE.get(node, "white")
    for line in update.get("log", []):
        console.print(f"[{style}]{node:<16}[/{style}] {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent", description="LangGraph data-analysis agent (Ollama or OpenAI)."
    )
    parser.add_argument("--data", "-d", help="Path to csv/tsv/json/xlsx/parquet dataset.")
    parser.add_argument("--prompt", "-p", help="The analysis question to answer.")
    parser.add_argument("--provider", choices=["ollama", "openai"],
                        help="Override LLM_PROVIDER for this run.")
    parser.add_argument("--model", help="Override the model name for this run.")
    parser.add_argument("--retries", type=int, help="Override MAX_ANALYSIS_RETRIES.")
    parser.add_argument("--check", action="store_true",
                        help="Probe the configured LLM backend and exit.")
    parser.add_argument("--show-report", action="store_true",
                        help="Render the finished report in the terminal.")
    args = parser.parse_args(argv)

    if args.provider:
        settings.provider = args.provider
    if args.model:
        if settings.provider == "ollama":
            settings.ollama_model = args.model
        else:
            settings.openai_model = args.model
    if args.retries is not None:
        settings.max_analysis_retries = args.retries

    if args.check:
        ok, message = health_check(settings)
        console.print(("[green]OK[/green]  " if ok else "[red]FAIL[/red] ") + message)
        return 0 if ok else 1

    if not args.data or not args.prompt:
        parser.error("--data and --prompt are required (or use --check).")
    if not Path(args.data).exists():
        console.print(f"[red]Dataset not found:[/red] {args.data}")
        return 1

    try:
        settings.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 1

    console.print(Panel.fit(
        f"[bold]{args.prompt}[/bold]\n"
        f"dataset : {args.data}\n"
        f"backend : {settings.provider} / {settings.model_name()}",
        title="AI Data Analyst", border_style="blue",
    ))

    from .graph import run_agent  # imported late so --check stays fast

    state = run_agent(args.data, args.prompt, on_event=_on_event)

    report_path = state.get("report_path")
    if not report_path:
        console.print("[red]The run finished without producing a report.[/red]")
        return 1

    console.print(f"\n[bold green]Report:[/bold green] {report_path}")
    if state.get("figures"):
        console.print(f"[bold green]Figures:[/bold green] {len(state['figures'])} in "
                      f"{Path(report_path).parent / 'figures'}")
    if args.show_report:
        console.print(Markdown(state.get("report_md", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
