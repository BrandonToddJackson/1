"""Clip + Transcript -> per-platform list[Post].

Zero-key default is deterministic string templates (see PLATFORM_RULES and
_template_post). If an LLM client is configured, tries the LLM path for
richer, on-voice copy first; any failure (bad JSON, a network error, an
empty result) falls back to templates with a logged warning, and any
platform the LLM's response is simply missing gets backfilled with a
template post rather than silently dropped. Same output contract either
way (mirrors skills/content-repurposer/SKILL.md).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from pipeline.llm import LLMClient, LLMResponseError, get_llm_client
from pipeline.schemas import Clip, DEFAULT_PLATFORMS, Learnings, Platform, Post, Transcript
from pipeline.textutils import truncate

log = logging.getLogger(__name__)

# Back-compat alias: this used to be defined here before moving to
# pipeline.textutils (shared with publisher.py). Kept so existing callers/
# tests referencing repurposer._truncate keep working.
_truncate = truncate


@dataclass(frozen=True)
class _PlatformRule:
    max_chars: int
    hashtag_count: int
    cta: str | None
    tone_note: str
    # The platform's real hard character cap once CTA+hashtags are appended
    # into a single caption (max_chars above is a soft budget on the *post
    # text alone*). None where the platform has no meaningful hard cap
    # (shorts is title/description metadata, newsletter is a full email
    # body). Enforced by publisher.build_caption, not here -- repurposer's
    # job is only to produce text within its own max_chars.
    hard_limit: int | None = None


PLATFORM_RULES: dict[Platform, _PlatformRule] = {
    "linkedin": _PlatformRule(
        max_chars=1200,
        hashtag_count=4,
        cta="What's been your experience with this? Curious to hear below.",
        tone_note="professional, first-person, short paragraphs",
        hard_limit=3000,
    ),
    "x": _PlatformRule(
        max_chars=260,
        hashtag_count=2,
        cta=None,
        tone_note="punchy, single idea, no fluff",
        hard_limit=280,
    ),
    "threads": _PlatformRule(
        max_chars=450,
        hashtag_count=3,
        cta="Curious what you think \U0001f447",
        tone_note="casual, conversational",
        hard_limit=500,
    ),
    "instagram": _PlatformRule(
        max_chars=2000,
        hashtag_count=8,
        cta="Save this for later ✨",
        tone_note="caption + hashtag block",
        hard_limit=2200,
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
    """Zero-key default: one templated Post per platform. Tries the LLM path
    first if pipeline.llm.get_llm_client() returns a configured client,
    falling back to templates (per-missing-platform, and entirely on total
    failure) rather than crashing."""
    client = get_llm_client()
    if client is not None:
        try:
            posts = _llm_posts(client, clip, transcript, platforms, learnings)
        except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure degrades gracefully
            log.warning("LLM post generation failed (%s); falling back to templates", exc)
        else:
            covered = {p.platform for p in posts}
            missing = [p for p in platforms if p not in covered]
            if missing:
                log.warning("LLM response missing platform(s) %s; backfilling with templates", missing)
                posts = posts + [_template_post(p, clip, transcript, learnings) for p in missing]
            if posts:
                return posts
            log.warning("LLM post generation returned no usable posts; falling back to templates")
    return [_template_post(p, clip, transcript, learnings) for p in platforms]


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


def _template_post(platform: Platform, clip: Clip, transcript: Transcript, learnings: Learnings | None) -> Post:
    rule = PLATFORM_RULES[platform]
    hook = clip.hook.rstrip(".").strip()
    body = _clip_text(clip, transcript)
    hashtags = _hashtags(clip, rule.hashtag_count, learnings)

    if platform == "shorts":
        title = truncate(hook, 100)
        prefix = f"Title: {title}\n\nDescription: "
        description_budget = rule.max_chars - len(prefix)
        description = truncate(body, description_budget)
        text = truncate(prefix + description, rule.max_chars)
        return Post(platform=platform, clip_id=clip.id, text=text, hashtags=hashtags, cta=rule.cta, generation_method="template")

    if platform == "newsletter":
        subject = truncate(hook, 80)
        text = f"Subject: {subject}\n\n{body}"
        text = truncate(text, rule.max_chars)
        return Post(platform=platform, clip_id=clip.id, text=text, hashtags=hashtags, cta=rule.cta, generation_method="template")

    pieces = [f"{hook}."]
    if body and body.lower() != hook.lower():
        pieces.append(body)
    if rule.cta:
        pieces.append(rule.cta)
    text = "\n\n".join(pieces)
    text = truncate(text, rule.max_chars)

    return Post(
        platform=platform,
        clip_id=clip.id,
        text=text,
        hashtags=hashtags,
        cta=rule.cta,
        generation_method="template",
    )


def _llm_posts(
    client: LLMClient,
    clip: Clip,
    transcript: Transcript,
    platforms: tuple[Platform, ...],
    learnings: Learnings | None,
) -> list[Post]:
    """LLM path: give the model the clip + surrounding transcript text, force
    structured JSON with one entry per requested platform (see
    skills/content-repurposer/SKILL.md). Every entry is validated -- an
    unknown platform or a missing text field is skipped with a warning, and
    text is truncated to that platform's max_chars regardless of what the
    model returned (an LLM post is not otherwise length-bounded)."""
    system = (
        "You are a social media repurposing assistant. Given a video clip's hook "
        "and transcript text, write one platform-native post per requested "
        "platform. Respect each platform's conventions (length, tone, hashtag "
        "use). Platforms requested: " + ", ".join(platforms)
    )
    if learnings:
        hints = []
        if learnings.top_keywords:
            hints.append(f"Known high-performing keywords: {', '.join(learnings.top_keywords)}.")
        if learnings.best_platforms:
            hints.append(f"Platforms that performed best previously: {', '.join(learnings.best_platforms)}.")
        if hints:
            system = system + " " + " ".join(hints)

    body = _clip_text(clip, transcript)
    user = f"Hook: {clip.hook}\nTopic: {clip.topic}\nTranscript excerpt: {body}"
    schema_hint = (
        '{"posts": [{"platform": str, "text": str, "hashtags": [str], "cta": str|null}]}'
    )
    result = client.complete_json(system=system, user=user, schema_hint=schema_hint)

    raw_posts = result.get("posts")
    if not isinstance(raw_posts, list):
        raise LLMResponseError("LLM post response missing a 'posts' list")

    posts: list[Post] = []
    for raw in raw_posts:
        if not isinstance(raw, dict):
            log.warning("LLM post entry is not an object, skipping: %r", raw)
            continue
        platform = raw.get("platform")
        if platform not in PLATFORM_RULES:
            log.warning("LLM returned unknown platform %r, skipping", platform)
            continue
        rule = PLATFORM_RULES[platform]
        text = truncate(str(raw.get("text", "")), rule.max_chars)
        posts.append(
            Post(
                platform=platform,
                clip_id=clip.id,
                text=text,
                hashtags=raw.get("hashtags") or [],
                cta=raw.get("cta"),
                generation_method="llm",
            )
        )
    return posts
