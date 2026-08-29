"""PerformanceRecord[] -> Learnings.

Always local/statistical -- there is no LLM path for this stage in v1 (see
skills/content-analyst/SKILL.md). A human (or a future platform-analytics
integration) supplies simple performance data; this module ranks it and
extracts winning keywords/hooks/platforms/lengths, which clip_selector and
repurposer then consume on the next run.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pipeline.clip_selector import STOPWORDS
from pipeline.schemas import Clip, Learnings, PerformanceRecord, Post

DEFAULT_LEARNINGS_PATH = Path("learnings.json")


def load_performance(path: Path) -> list[PerformanceRecord]:
    """Reads a CSV with columns: post_id, platform, clip_id, views, likes,
    comments, shares, engagement_rate (last one optional)."""
    records: list[PerformanceRecord] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            engagement_rate = row.get("engagement_rate") or None
            records.append(
                PerformanceRecord(
                    post_id=row["post_id"],
                    platform=row["platform"],
                    clip_id=row.get("clip_id") or None,
                    views=int(row.get("views") or 0),
                    likes=int(row.get("likes") or 0),
                    comments=int(row.get("comments") or 0),
                    shares=int(row.get("shares") or 0),
                    engagement_rate=float(engagement_rate) if engagement_rate else None,
                )
            )
    return records


def analyze(
    records: list[PerformanceRecord],
    posts: list[Post] | None = None,
    clips: list[Clip] | None = None,
    top_fraction: float = 0.25,
) -> Learnings:
    """Ranks records by engagement rate, extracts what the top quartile has
    in common, and returns it as Learnings for the next run to consume."""
    if not records:
        return Learnings(notes="No performance data yet -- run analyze again after your first batch posts.")

    ranked = sorted(records, key=lambda r: r.compute_engagement_rate(), reverse=True)
    top_n = max(1, round(len(ranked) * top_fraction))
    top = ranked[:top_n]
    top_clip_ids = {r.clip_id for r in top if r.clip_id}

    best_platforms = _rank_platforms(records)

    clip_by_id = {c.id: c for c in (clips or [])}
    top_clips = [clip_by_id[cid] for cid in top_clip_ids if cid in clip_by_id]

    top_keywords = _extract_keywords(top_clips, posts, top_clip_ids)
    top_hook_patterns = [c.hook for c in top_clips][:10]

    ideal_range = None
    if top_clips:
        durations = sorted(c.duration for c in top_clips)
        ideal_range = (durations[0], durations[-1])

    notes = (
        f"Derived from {len(records)} performance record(s); "
        f"top {len(top)} by engagement rate drove these learnings."
    )

    return Learnings(
        top_keywords=top_keywords,
        top_hook_patterns=top_hook_patterns,
        best_platforms=best_platforms,
        ideal_clip_length_range=ideal_range,
        notes=notes,
    )


def _rank_platforms(records: list[PerformanceRecord]) -> list[str]:
    scores: dict[str, list[float]] = {}
    for r in records:
        scores.setdefault(r.platform, []).append(r.compute_engagement_rate())
    return sorted(scores, key=lambda p: sum(scores[p]) / len(scores[p]), reverse=True)


def _extract_keywords(top_clips: list[Clip], posts: list[Post] | None, top_clip_ids: set[str]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    # Hashtags from top-performing posts are an explicit, human-curated
    # keyword signal -- prefer them over words mined from clip text.
    for post in posts or []:
        if post.clip_id not in top_clip_ids:
            continue
        for tag in post.hashtags:
            slug = tag.lstrip("#").lower()
            if slug and slug not in seen:
                seen.add(slug)
                keywords.append(slug)

    counts: Counter[str] = Counter()
    for clip in top_clips:
        tokens = re.findall(r"[a-zA-Z']+", f"{clip.hook} {clip.topic}".lower())
        for t in tokens:
            if t in STOPWORDS or len(t) < 3:
                continue
            counts[t] += 1
    for word, _ in counts.most_common(10):
        if word not in seen:
            seen.add(word)
            keywords.append(word)

    return keywords[:10]


def save_learnings(learnings: Learnings, path: Path = DEFAULT_LEARNINGS_PATH) -> None:
    path.write_text(learnings.model_dump_json(indent=2), encoding="utf-8")


def load_learnings(path: Path = DEFAULT_LEARNINGS_PATH) -> Learnings | None:
    if not path.exists():
        return None
    return Learnings.model_validate_json(path.read_text(encoding="utf-8"))
