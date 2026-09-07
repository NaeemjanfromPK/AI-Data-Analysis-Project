"""Central runtime configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Everything the agent needs to know about *how* to run."""

    # --- provider selection -------------------------------------------------
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())

    # Ollama (local)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    )

    # OpenAI (API)
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))

    # --- generation ---------------------------------------------------------
    temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.1))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 180))

    # --- execution ----------------------------------------------------------
    max_analysis_retries: int = field(default_factory=lambda: _int("MAX_ANALYSIS_RETRIES", 3))
    sandbox_timeout: int = field(default_factory=lambda: _int("SANDBOX_TIMEOUT", 120))
    runs_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("RUNS_DIR", "runs")
    )

    def model_name(self) -> str:
        return self.ollama_model if self.provider == "ollama" else self.openai_model

    def validate(self) -> None:
        if self.provider not in {"ollama", "openai"}:
            raise ValueError(
                f"LLM_PROVIDER must be 'ollama' or 'openai', got {self.provider!r}"
            )
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is unset. "
                "Put it in .env or switch to LLM_PROVIDER=ollama."
            )


settings = Settings()
