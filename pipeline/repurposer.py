"""Clip + Transcript -> per-platform list[Post].

Zero-key default is deterministic string templates (see PLATFORM_RULES and
_template_post). If an LLM client is configured, delegates to the LLM path
for richer, on-voice copy -- same output contract either way (mirrors
skills/content-repurposer/SKILL.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.llm import LLMClient, get_llm_client
from pipeline.schemas import Clip, DEFAULT_PLATFORMS, Learnings, Platform, Post, Transcript


@dataclass(frozen=True)
class _PlatformRule:
    max_chars: int
    hashtag_count: int
    cta: str | None
    tone_note: str


PLATFORM_RULES: dict[Platform, _PlatformRule] = {
    "linkedin": _PlatformRule(
        max_chars=1200,
        hashtag_count=4,
        cta="What's been your experience with this? Curious to hear below.",
        tone_note="professional, first-person, short paragraphs",
    ),
    "x": _PlatformRule(
        max_chars=260,
        hashtag_count=2,
        cta=None,
        tone_note="punchy, single idea, no fluff",
    ),
    "threads": _PlatformRule(
        max_chars=450,
        hashtag_count=3,
        cta="Curious what you think \U0001f447",
        tone_note="casual, conversational",
    ),
    "instagram": _PlatformRule(
        max_chars=2000,
        hashtag_count=8,
        cta="Save this for later ✨",
        tone_note="caption + hashtag block",
    ),
    "shorts": _PlatformRule(
        max_chars=480,
        hashtag_count=3,
        cta=None,
        tone_note="Title: ... / Description: ... metadata, not prose",
    ),
    "newsletter": _PlatformRule(
        max_chars=2500,
        hashtag_count=0,
        cta="Reply and tell me what resonated most.",
        tone_note="longer-form, subject line + body",
    ),
}


def generate_posts(
    clip: Clip,
    transcript: Transcript,
    platforms: tuple[Platform, ...] = DEFAULT_PLATFORMS,
    learnings: Learnings | None = None,
) -> list[Post]:
    """Zero-key default: one templated Post per platform. LLM path used only
    if pipeline.llm.get_llm_client() returns a configured client."""
    client = get_llm_client()
    if client is None:
        return [_template_post(p, clip, transcript, learnings) for p in platforms]
    return _llm_posts(client, clip, transcript, platforms)


def _clip_text(clip: Clip, transcript: Transcript) -> str:
    segs = [seg for seg in transcript.segments if seg.id in clip.source_segment_ids]
    segs.sort(key=lambda s: s.id)
    text = " ".join(seg.text.strip() for seg in segs if seg.text.strip())
    return text or clip.hook


def _hashtags(clip: Clip, count: int, learnings: Learnings | None) -> list[str]:
    if count == 0:
        return []
    words: list[str] = []
    if learnings and learnings.top_keywords:
        words.extend(learnings.top_keywords)
    words.extend(clip.topic.split())
    tags: list[str] = []
    seen: set[str] = set()
    for w in words:
        slug = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        tags.append("#" + slug)
        if len(tags) >= count:
            break
    return tags


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _template_post(platform: Platform, clip: Clip, transcript: Transcript, learnings: Learnings | None) -> Post:
    rule = PLATFORM_RULES[platform]
    hook = clip.hook.rstrip(".").strip()
    body = _clip_text(clip, transcript)
    hashtags = _hashtags(clip, rule.hashtag_count, learnings)

    if platform == "shorts":
        title = _truncate(hook, 100)
        description = _truncate(body, rule.max_chars - len(title) - 20)
        text = f"Title: {title}\n\nDescription: {description}"
        return Post(platform=platform, clip_id=clip.id, text=text, hashtags=hashtags, cta=rule.cta, generation_method="template")

    if platform == "newsletter":
        subject = _truncate(hook, 80)
        text = f"Subject: {subject}\n\n{body}"
        text = _truncate(text, rule.max_chars)
        return Post(platform=platform, clip_id=clip.id, text=text, hashtags=hashtags, cta=rule.cta, generation_method="template")

    pieces = [f"{hook}."]
    if body and body.lower() != hook.lower():
        pieces.append(body)
    if rule.cta:
        pieces.append(rule.cta)
    text = "\n\n".join(pieces)
    text = _truncate(text, rule.max_chars)

    return Post(
        platform=platform,
        clip_id=clip.id,
        text=text,
        hashtags=hashtags,
        cta=rule.cta,
        generation_method="template",
    )


def _llm_posts(client: LLMClient, clip: Clip, transcript: Transcript, platforms: tuple[Platform, ...]) -> list[Post]:
    """LLM path: give the model the clip + surrounding transcript text, force
    structured JSON with one entry per requested platform (see
    skills/content-repurposer/SKILL.md)."""
    system = (
        "You are a social media repurposing assistant. Given a video clip's hook "
        "and transcript text, write one platform-native post per requested "
        "platform. Respect each platform's conventions (length, tone, hashtag "
        "use). Platforms requested: " + ", ".join(platforms)
    )
    body = _clip_text(clip, transcript)
    user = f"Hook: {clip.hook}\nTopic: {clip.topic}\nTranscript excerpt: {body}"
    schema_hint = (
        '{"posts": [{"platform": str, "text": str, "hashtags": [str], "cta": str|null}]}'
    )
    result = client.complete_json(system=system, user=user, schema_hint=schema_hint)

    posts: list[Post] = []
    for raw in result.get("posts", []):
        platform = raw.get("platform")
        if platform not in PLATFORM_RULES:
            continue
        posts.append(
            Post(
                platform=platform,
                clip_id=clip.id,
                text=raw.get("text", ""),
                hashtags=raw.get("hashtags", []) or [],
                cta=raw.get("cta"),
                generation_method="llm",
            )
        )
    return posts
