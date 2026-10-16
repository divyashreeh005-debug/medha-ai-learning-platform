"""LLM runtime client and detector for Qwen3-8B-Instruct (Phase 4).

Connects to local inference runtimes (LM Studio or Ollama) using an
OpenAI-compatible chat completions interface.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.config import settings

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Raised when communication with the LLM runtime fails."""


def resolve_model_id(model_type: str = "qwen", available_models: list[str] | None = None) -> Optional[str]:
    """Resolve the active model ID in LM Studio for either 'qwen' or 'gemma'.
    
    Dynamically checks available model IDs so slight naming differences
    (e.g., 'gemma-4-e2b-it-qat' vs 'google/gemma-4-E2B-it-qat-q4_0-gguf')
    resolve transparently.
    """
    models = available_models
    if models is None:
        try:
            resp = requests.get(f"{settings.llm_base_url}/models", timeout=2)
            if resp.status_code == 200:
                models = [m.get("id", "") for m in resp.json().get("data", [])]
        except Exception:
            models = []

    if not models:
        return None

    if model_type == "gemma":
        # Check configured preferences
        candidates = [settings.gemma_model, settings.gemma_model_alias, "gemma-4", "gemma"]
        for cand in candidates:
            cand_lower = cand.lower()
            for m in models:
                if cand_lower == m.lower() or cand_lower in m.lower():
                    return m
        return None

    if model_type in ("qwen", "reasoning", "default"):
        # Default to Qwen reasoning model
        candidates = [settings.qwen_model, settings.llm_model, "qwen3", "qwen"]
        for cand in candidates:
            cand_lower = cand.lower()
            for m in models:
                if cand_lower == m.lower() or cand_lower in m.lower():
                    return m
        return None

    return None


def check_llm_runtime() -> dict[str, Any]:
    """Detect whether LM Studio or Ollama is running and responsive, reporting Qwen and Gemma status."""
    # 1. Check configured base_url (LM Studio)
    try:
        resp = requests.get(f"{settings.llm_base_url}/models", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            qwen_id = resolve_model_id("qwen", available_models=models)
            gemma_id = resolve_model_id("gemma", available_models=models)

            return {
                "available": bool(qwen_id or gemma_id),
                "runtime": "LM Studio" if "1234" in settings.llm_base_url else "OpenAI-compatible local",
                "base_url": settings.llm_base_url,
                "model": qwen_id or settings.llm_model,
                "qwen_available": bool(qwen_id),
                "gemma_available": bool(gemma_id),
                "qwen_model": qwen_id,
                "gemma_model": gemma_id,
                "available_models": models,
                "details": f"Online with {len(models)} model(s) available: {', '.join(models)}",
            }
    except Exception:
        pass

    # 2. Check Ollama fallback (port 11434)
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {
                "available": True,
                "runtime": "Ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3:8b",
                "qwen_available": True,
                "gemma_available": False,
                "qwen_model": "qwen3:8b",
                "gemma_model": None,
                "available_models": models,
                "details": f"Online with {len(models)} model(s) available: {', '.join(models)}",
            }
    except Exception:
        pass

    return {
        "available": False,
        "runtime": "Unavailable",
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "qwen_available": False,
        "gemma_available": False,
        "qwen_model": None,
        "gemma_model": None,
        "available_models": [],
        "details": "Local LLM server is not running. Start LM Studio (port 1234) or Ollama (port 11434).",
    }


def call_llm_chat(
    messages: list[dict[str, str]],
    max_tokens: int = 800,
    temperature: float | None = None,
    model: str | None = None,
) -> str:
    """Send a chat completion request to the local LLM and return the assistant text response.
    
    Supports targeting either Qwen (reasoning) or Gemma (translation) locally.
    """
    endpoint = f"{settings.llm_base_url}/chat/completions"
    temp = settings.llm_temperature if temperature is None else temperature

    chosen_model = model or resolve_model_id("qwen") or settings.llm_model

    payload = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=settings.llm_timeout_seconds,
        )
    except requests.exceptions.ConnectionError as exc:
        raise LLMClientError(
            f"Cannot connect to LLM server at {settings.llm_base_url}. Ensure LM Studio or Ollama is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMClientError(
            f"LLM request timed out after {settings.llm_timeout_seconds} seconds."
        ) from exc
    except requests.RequestException as exc:
        raise LLMClientError(f"HTTP error communicating with LLM server: {exc}") from exc

    if response.status_code != 200:
        raise LLMClientError(
            f"LLM server returned HTTP {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMClientError("LLM response contained no choices.")

        message = choices[0].get("message", {})
        content = message.get("content") or ""

        # For reasoning models where reasoning tokens are populated
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]

        if not content.strip():
            raise LLMClientError("LLM returned an empty content string.")

        return content.strip()
    except Exception as exc:
        if isinstance(exc, LLMClientError):
            raise
        raise LLMClientError(f"Failed to parse LLM response JSON: {exc}") from exc
