"""list[Post] (+ optional finished clip media) -> list[PublishResult].

Default: writes a local "outbox" package per post that a human can review
and manually upload -- this is the REVIEW step in the pipeline diagram.
Live publish is attempted only if a provider key is configured, preferring
Blotato over Ayrshare when both are set (Blotato's entry tier is ~$120/mo
cheaper for a comparable solo-operator footprint, with a real REST API).

IMPORTANT LIMITATION: neither live provider can receive local clip media.
Both Ayrshare's `mediaUrls` and Blotato's `/v2/media` endpoint require a
publicly-hosted URL, not a local file path -- attaching real video would
need a separate media-hosting/upload step this pipeline doesn't implement.
Live publish here is text-only; anything needing the actual video clip
still goes through the outbox for manual upload.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from types import ModuleType

from pipeline.config import get_settings
from pipeline.repurposer import PLATFORM_RULES
from pipeline.schemas import Platform, Post, PublishResult
from pipeline.storage import run_dir
from pipeline.textutils import truncate

log = logging.getLogger(__name__)

# Internal platform name -> what each provider's API expects. Neither
# provider has a "shorts" or "newsletter" concept: shorts maps to the best
# available fit for vertical video (youtube); newsletter has no live target
# at all for either provider and always routes to the outbox regardless of
# which key is configured.
_AYRSHARE_PLATFORMS: dict[Platform, str] = {
    "linkedin": "linkedin",
    "x": "twitter",
    "threads": "threads",
    "instagram": "instagram",
    "shorts": "youtube",
}

_BLOTATO_PLATFORMS: dict[Platform, str] = {
    "linkedin": "linkedin",
    "x": "twitter",
    "threads": "threads",
    "instagram": "instagram",
    "shorts": "youtube",
}


def _get_requests() -> ModuleType | None:
    """Lazy, test-friendly import seam. `requests` is only needed on the
    live-publish paths (the `publish` extra, not a base dependency) and
    isn't installed by default -- returns None instead of raising so
    callers report a clean per-post failure rather than crashing the whole
    batch, and so tests can monkeypatch this without requests installed."""
    try:
        import requests
    except ImportError:
        return None
    return requests


def build_caption(post: Post) -> str:
    """Assembles the single caption (text + CTA + hashtags) used by the
    outbox AND both live-publish paths -- one shared place, so "live publish
    drops the CTA/hashtags the outbox includes" can't happen again. If the
    platform has a hard_limit (see repurposer.PLATFORM_RULES) and the full
    assembly would exceed it, shrinks the *text* portion first so the
    CTA+hashtag block survives; only truncates the whole string if the
    CTA+hashtags alone already overflow the limit."""
    rule = PLATFORM_RULES.get(post.platform)
    hard_limit = rule.hard_limit if rule else None

    suffix_parts = []
    if post.cta and post.cta not in post.text:
        suffix_parts.append(post.cta)
    if post.hashtags:
        suffix_parts.append(" ".join(post.hashtags))
    suffix = ("\n\n" + "\n\n".join(suffix_parts)) if suffix_parts else ""

    text = post.text
    if hard_limit is not None and len(text) + len(suffix) > hard_limit:
        budget = hard_limit - len(suffix)
        if budget <= 0:
            return truncate(text + suffix, hard_limit)
        text = truncate(text, budget)

    return text + suffix


def publish(
    run_id: str,
    posts: list[Post],
    clip_media: dict[str, Path] | None = None,
) -> list[PublishResult]:
    """Zero-key default: write a local outbox. Live publish (Blotato
    preferred, then Ayrshare) only if a key is configured. A platform
    neither provider can reach (newsletter) always routes to the outbox,
    even when a live key is set."""
    settings = get_settings()
    clip_media = clip_media or {}

    if settings.blotato_api_key:
        return _publish_with_provider(
            run_id,
            posts,
            clip_media,
            platform_map=_BLOTATO_PLATFORMS,
            live_fn=lambda live_posts: _publish_blotato(settings.blotato_api_key, settings.blotato_account_ids, live_posts),
        )
    if settings.ayrshare_api_key:
        return _publish_with_provider(
            run_id,
            posts,
            clip_media,
            platform_map=_AYRSHARE_PLATFORMS,
            live_fn=lambda live_posts: _publish_ayrshare(settings.ayrshare_api_key, live_posts),
        )
    return _write_outbox(run_id, posts, clip_media)


def _publish_with_provider(run_id, posts, clip_media, platform_map, live_fn) -> list[PublishResult]:
    """Splits posts into "this provider can publish it live" vs "route to
    the outbox" (unsupported platforms), publishes the live subset via
    live_fn, writes the rest to the outbox, and returns results in the
    original post order."""
    live_posts = [p for p in posts if p.platform in platform_map]
    outbox_posts = [p for p in posts if p.platform not in platform_map]

    live_results = live_fn(live_posts) if live_posts else []
    outbox_results = _write_outbox(run_id, outbox_posts, clip_media) if outbox_posts else []

    live_iter = iter(live_results)
    outbox_iter = iter(outbox_results)
    results: list[PublishResult] = []
    for post in posts:
        results.append(next(live_iter) if post.platform in platform_map else next(outbox_iter))
    return results


def _write_outbox(run_id: str, posts: list[Post], clip_media: dict[str, Path]) -> list[PublishResult]:
    results: list[PublishResult] = []
    for post in posts:
        clip_label = post.clip_id or "no-clip"
        post_dir = run_dir(run_id) / "outbox" / clip_label / post.platform
        post_dir.mkdir(parents=True, exist_ok=True)

        (post_dir / "caption.txt").write_text(build_caption(post) + "\n", encoding="utf-8")
        (post_dir / "metadata.json").write_text(post.model_dump_json(indent=2), encoding="utf-8")

        media_path = clip_media.get(post.clip_id) if post.clip_id else None
        if media_path and Path(media_path).exists():
            dest = post_dir / f"media{Path(media_path).suffix}"
            shutil.copy2(media_path, dest)

        results.append(
            PublishResult(
                platform=post.platform,
                method="outbox",
                location=str(post_dir),
                status="ready",
            )
        )
    return results


def _publish_ayrshare(api_key: str, posts: list[Post]) -> list[PublishResult]:
    """Live publish via Ayrshare (https://www.ayrshare.com/docs/apis/post/overview).
    `posts` is already filtered to platforms Ayrshare supports (see
    _AYRSHARE_PLATFORMS) -- no per-post platform check needed here."""
    requests = _get_requests()
    if requests is None:
        return [
            PublishResult(platform=p.platform, method="ayrshare", status="failed", location="requests not installed - pip install -e '.[publish]'")
            for p in posts
        ]

    results: list[PublishResult] = []
    for post in posts:
        payload = {
            "post": build_caption(post),
            "platforms": [_AYRSHARE_PLATFORMS[post.platform]],
        }
        try:
            resp = requests.post(
                "https://app.ayrshare.com/api/post",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results.append(
                PublishResult(
                    platform=post.platform,
                    method="ayrshare",
                    location=str(data.get("id", json.dumps(data))),
                    status="published",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the batch
            results.append(PublishResult(platform=post.platform, method="ayrshare", location=str(exc), status="failed"))
    return results


def _publish_blotato(api_key: str, account_ids: dict[str, str], posts: list[Post]) -> list[PublishResult]:
    """Live publish via Blotato (https://backend.blotato.com/v2/posts,
    header `blotato-api-key`). `posts` is already filtered to platforms
    Blotato supports (see _BLOTATO_PLATFORMS).

    UNVERIFIED: the exact request/response shape below was reconstructed
    from Blotato help-doc search-result snippets -- help.blotato.com was
    egress-blocked during implementation, so the real payload shape was
    never directly confirmed. Verify against a real key and
    https://help.blotato.com/api/start (and /api/api-reference/publish-post)
    before relying on this in production.
    """
    requests = _get_requests()
    if requests is None:
        return [
            PublishResult(platform=p.platform, method="blotato", status="failed", location="requests not installed - pip install -e '.[publish]'")
            for p in posts
        ]

    results: list[PublishResult] = []
    for post in posts:
        target_platform = _BLOTATO_PLATFORMS[post.platform]
        account_id = account_ids.get(post.platform)
        if not account_id:
            results.append(
                PublishResult(
                    platform=post.platform,
                    method="blotato",
                    status="failed",
                    location=f"no BLOTATO_ACCOUNT_IDS entry for platform '{post.platform}' (set it as JSON, e.g. {{\"{post.platform}\": \"acct_...\"}})",
                )
            )
            continue
        payload = {
            "post": {
                "accountId": account_id,
                "content": {"text": build_caption(post), "mediaUrls": [], "platform": target_platform},
                "target": {"targetType": target_platform},
            }
        }
        try:
            resp = requests.post(
                "https://backend.blotato.com/v2/posts",
                headers={"blotato-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results.append(
                PublishResult(
                    platform=post.platform,
                    method="blotato",
                    location=str(data.get("id", json.dumps(data))),
                    status="published",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the batch
            results.append(PublishResult(platform=post.platform, method="blotato", location=str(exc), status="failed"))
    return results
