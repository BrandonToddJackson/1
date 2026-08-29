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
    comments, shares, engagement_rate. Only post_id/platform are required --
    the rest default to 0/None when absent. Tolerates the messiness of real
    exports (Excel/Sheets BOM, thousands separators, "500.0"-style floats,
    a trailing % on engagement_rate, extra whitespace in headers/values) but
    fails loudly -- naming the file/row/column -- on a value that's actually
    present and unparseable, rather than silently defaulting it to 0."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        headers = {(h or "").strip() for h in reader.fieldnames}
        missing = {"post_id", "platform"} - headers
        if missing:
            found = ", ".join(sorted(h for h in headers if h)) or "(none)"
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(sorted(missing))}; found columns: {found}"
            )

        records: list[PerformanceRecord] = []
        for line_no, raw_row in enumerate(reader, start=2):  # header is line 1
            row = {
                (k or "").strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in raw_row.items()
                if k is not None
            }
            post_id = row.get("post_id")
            platform = row.get("platform")
            if not post_id:
                raise ValueError(f"{path}: row {line_no}: missing post_id")
            if not platform:
                raise ValueError(f"{path}: row {line_no}: missing platform")
            try:
                records.append(
                    PerformanceRecord(
                        post_id=post_id,
                        platform=platform,
                        clip_id=row.get("clip_id") or None,
                        views=_parse_int(row.get("views"), "views"),
                        likes=_parse_int(row.get("likes"), "likes"),
                        comments=_parse_int(row.get("comments"), "comments"),
                        shares=_parse_int(row.get("shares"), "shares"),
                        engagement_rate=_parse_rate(row.get("engagement_rate")),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"{path}: row {line_no}: {exc}") from exc
    return records


def _parse_int(value: str | None, field: str) -> int:
    if value in (None, ""):
        return 0
    cleaned = value.replace(",", "").replace("_", "")
    try:
        return int(float(cleaned))
    except ValueError:
        raise ValueError(f"column '{field}' has non-numeric value {value!r}")


def _parse_rate(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    cleaned = value.strip()
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    cleaned = cleaned.replace(",", "")
    try:
        rate = float(cleaned)
    except ValueError:
        raise ValueError(f"column 'engagement_rate' has non-numeric value {value!r}")
    return rate / 100 if is_percent else rate


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
    durations = sorted(c.duration for c in top_clips if c.duration > 0)
    if durations:
        ideal_range = (durations[0], durations[-1])

    notes = (
        f"Derived from {len(records)} performance record(s); "
        f"top {len(top)} by engagement rate drove these learnings."
    )
    if top_clip_ids and not top_clips:
        notes += (
            f" Note: {len(top_clip_ids)} performance record(s) referenced clip_id(s) not found "
            "in the provided clips -- check that --run-id matches the run these posts came from."
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
