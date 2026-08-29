"""list[Post] (+ optional finished clip media) -> list[PublishResult].

Default: writes a local "outbox" package per post that a human can review
and manually upload -- this is the REVIEW step in the pipeline diagram.
Only if AYRSHARE_API_KEY is configured does this attempt a live publish via
the Ayrshare API instead.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas import Post, PublishResult
from pipeline.storage import run_dir


def publish(
    run_id: str,
    posts: list[Post],
    clip_media: dict[str, Path] | None = None,
) -> list[PublishResult]:
    """Zero-key default: write a local outbox. Live Ayrshare publish only if
    AYRSHARE_API_KEY is set."""
    settings = get_settings()
    if not settings.ayrshare_api_key:
        return _write_outbox(run_id, posts, clip_media or {})
    return _publish_ayrshare(settings.ayrshare_api_key, run_id, posts, clip_media or {})


def _write_outbox(run_id: str, posts: list[Post], clip_media: dict[str, Path]) -> list[PublishResult]:
    results: list[PublishResult] = []
    for post in posts:
        clip_label = post.clip_id or "no-clip"
        post_dir = run_dir(run_id) / "outbox" / clip_label / post.platform
        post_dir.mkdir(parents=True, exist_ok=True)

        caption_lines = [post.text]
        if post.cta and post.cta not in post.text:
            caption_lines.append(post.cta)
        if post.hashtags:
            caption_lines.append(" ".join(post.hashtags))
        (post_dir / "caption.txt").write_text("\n\n".join(caption_lines) + "\n", encoding="utf-8")

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


def _publish_ayrshare(
    api_key: str, run_id: str, posts: list[Post], clip_media: dict[str, Path]
) -> list[PublishResult]:
    """Live publish via Ayrshare (https://www.ayrshare.com/docs/apis/post/overview).

    Kept minimal and untested against the live API in v1 (requires a real
    key) -- structurally correct, lazy-imports `requests` so it never affects
    the zero-key install.
    """
    import requests  # lazy import: only needed on the live-publish path

    results: list[PublishResult] = []
    for post in posts:
        payload = {
            "post": post.text,
            "platforms": [post.platform],
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
                    location=data.get("id", json.dumps(data)),
                    status="published",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report the failure, don't crash the batch
            results.append(
                PublishResult(
                    platform=post.platform,
                    method="ayrshare",
                    location=str(exc),
                    status="failed",
                )
            )
    return results
