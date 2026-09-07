"""LangGraph data-analysis agent with a dual LLM backend (Ollama / OpenAI)."""
from .config import Settings, settings
from .graph import build_graph, run_agent
from .state import AgentState

__version__ = "0.1.0"
__all__ = ["Settings", "settings", "build_graph", "run_agent", "AgentState"]
