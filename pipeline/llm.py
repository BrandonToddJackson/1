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


class LLMResponseError(RuntimeError):
    """Raised when an LLM's response can't be parsed as the expected JSON
    object -- a markdown-fenced response, leading/trailing prose, a refusal,
    or a truncated/non-JSON reply. Callers (clip_selector, repurposer) catch
    this and fall back to the zero-key heuristic/template path rather than
    crashing the whole run over one bad LLM response."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        preview = (raw_text or "").strip()
        if len(preview) > 500:
            preview = preview[:500] + "... (truncated)"
        full = message + (f"\nraw response: {preview!r}" if preview else "")
        super().__init__(full)
        self.raw_text = raw_text


def _strip_markdown_fences(text: str) -> str:
    """Strips a leading/trailing ``` or ```json fence, if present. LLMs
    routinely wrap JSON in a fence even when explicitly asked not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_balanced_json_object(text: str) -> str | None:
    """Scans for the first balanced {...} substring, respecting string
    literals and escapes (so a brace inside a quoted string doesn't
    miscount depth). Handles the common "Here are the clips: {...}" case
    where a plain json.loads on the whole string fails. Returns None if no
    balanced object is found."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_response(text: str | None) -> dict:
    """The single place every provider's raw text passes through on its way
    to becoming the dict callers expect. Never trusts json.loads on raw LLM
    output alone -- strips fences first, and falls back to extracting a
    balanced JSON object from surrounding prose before giving up."""
    if not text or not text.strip():
        raise LLMResponseError("LLM returned an empty response", text or "")

    candidate = _strip_markdown_fences(text)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        extracted = _extract_balanced_json_object(candidate)
        if extracted is None:
            raise LLMResponseError("could not find valid JSON in the LLM response", text)
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM response was not valid JSON ({exc})", text) from exc

    if not isinstance(parsed, dict):
        raise LLMResponseError(f"expected a JSON object, got {type(parsed).__name__}", text)

    return parsed


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
        if not resp.choices:
            raise LLMResponseError("OpenAI returned no choices")
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            reason = getattr(choice, "finish_reason", "unknown")
            raise LLMResponseError(f"OpenAI returned no content (finish_reason={reason})")
        return _parse_json_response(content)


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
        if not text.strip():
            reason = getattr(resp, "stop_reason", "unknown")
            raise LLMResponseError(f"Anthropic returned no text content (stop_reason={reason})")
        return _parse_json_response(text)


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
