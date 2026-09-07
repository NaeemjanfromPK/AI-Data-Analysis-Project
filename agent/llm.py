"""Dual-backend LLM factory: local Ollama models, or the OpenAI API.

Both branches return a LangChain ``Runnable`` chat model so every node in the graph
is written once and stays provider-agnostic.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Settings, settings as default_settings

_CACHE: dict[tuple, Runnable] = {}


def build_llm(cfg: Settings | None = None, *, json_mode: bool = False) -> Runnable:
    """Return a chat model for the configured provider.

    Parameters
    ----------
    json_mode:
        Ask the backend to constrain output to valid JSON. Supported natively by
        both Ollama (``format="json"``) and OpenAI (``response_format``).
    """
    cfg = cfg or default_settings
    cfg.validate()

    key = (cfg.provider, cfg.model_name(), cfg.temperature, json_mode)
    if key in _CACHE:
        return _CACHE[key]

    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": cfg.ollama_model,
            "base_url": cfg.ollama_base_url,
            "temperature": cfg.temperature,
            "num_ctx": 8192,
        }
        if json_mode:
            kwargs["format"] = "json"
        llm: Runnable = ChatOllama(**kwargs)
    else:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": cfg.openai_model,
            "api_key": cfg.openai_api_key,
            "temperature": cfg.temperature,
            "timeout": cfg.llm_timeout,
            "max_retries": 2,
        }
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        llm = ChatOpenAI(**kwargs)
        if json_mode:
            # .bind() keeps this working across langchain-openai 0.2.x and 1.x,
            # where `model_kwargs` handling of response_format differs.
            llm = llm.bind(response_format={"type": "json_object"})

    _CACHE[key] = llm
    return llm


def chat(system: str, user: str, *, llm: Runnable | None = None, json_mode: bool = False) -> str:
    """One-shot system+user call returning plain text."""
    model = llm or build_llm(json_mode=json_mode)
    reply = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = reply.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content).strip()


# --------------------------------------------------------------------------
# Output parsing helpers — small local models are chatty, so be forgiving.
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the Python body out of a model reply, fenced or not."""
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply."""
    for candidate in (*_FENCE.findall(text), text):
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # last resort: first balanced {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def health_check(cfg: Settings | None = None) -> tuple[bool, str]:
    """Verify the selected backend actually answers. Returns (ok, message)."""
    cfg = cfg or default_settings
    try:
        cfg.validate()
        reply = chat("Reply with the single word: ready", "ping", llm=build_llm(cfg))
        return True, f"{cfg.provider}:{cfg.model_name()} -> {reply[:60]!r}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        return False, f"{cfg.provider}:{cfg.model_name()} unavailable -> {exc}"
