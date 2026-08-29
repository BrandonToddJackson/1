"""Small text helpers shared across stages that assemble platform copy."""

from __future__ import annotations


def truncate(text: str, max_chars: int) -> str:
    """Truncates on a word boundary, appending an ellipsis. Guards
    max_chars <= 0 (unreachable via the current PLATFORM_RULES, but a
    future rule change could hit it -- without the guard, a negative slice
    count returns nearly the whole string instead of failing safely)."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"
