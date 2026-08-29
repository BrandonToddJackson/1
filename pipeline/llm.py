"""The one shared seam between "zero-key heuristic" and "LLM-enhanced".

``get_llm_client()`` is the single decision point every LLM-optional module
(clip_selector, repurposer) calls. It returns ``None`` when no API key is
configured -- callers MUST treat that as "use the deterministic fallback".

The SDKs (``openai``, ``anthropic``) are imported lazily, inside the client
classes' ``__init__``, so the base install (no ``[llm]`` extra) never needs
them installed and never pays their import cost when no key is set.
"""

from __future__ import annotations

import json
from typing import Protocol

from pipeline.config import get_settings


class LLMClient(Protocol):
    """Minimal contract: send a system+user prompt, get back parsed JSON.

    Every LLM-enabled stage asks for strict JSON output (schema described in
    the prompt) and parses it with this single method, so swapping providers
    never touches callers.
    """

    def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        ...


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # lazy import: optional dependency

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{system}\n\nRespond ONLY with JSON matching: {schema_hint}"},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content)


class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        import anthropic  # lazy import: optional dependency

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=f"{system}\n\nRespond ONLY with valid JSON matching: {schema_hint}. No prose, no markdown fences.",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return json.loads(text)


def get_llm_client() -> LLMClient | None:
    """Anthropic is preferred if both keys are set (matches .env.example).

    Returns None with zero keys configured -- this IS the "works without API
    keys" contract, not an error path.
    """
    settings = get_settings()
    if settings.anthropic_api_key:
        return AnthropicClient(settings.anthropic_api_key)
    if settings.openai_api_key:
        return OpenAIClient(settings.openai_api_key)
    return None
