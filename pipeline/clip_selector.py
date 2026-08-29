"""Transcript -> ranked list[Clip].

Zero-key default is a heuristic scorer: candidate windows are snapped to
pause/sentence boundaries in the word-level timestamps, scored on hook-word
density and closeness to an ideal length, then de-overlapped (non-max
suppression) down to ``max_clips``. If ``pipeline.llm.get_llm_client()``
returns a client, the LLM path is used instead -- same output contract
either way (mirrors skills/clip-selector/SKILL.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.llm import LLMClient, get_llm_client
from pipeline.schemas import Clip, Learnings, Transcript, Word

DEFAULT_MAX_CLIPS = 5
DEFAULT_MIN_LEN = 20.0
DEFAULT_MAX_LEN = 90.0

# A gap this long (seconds) between two consecutive words is treated as a
# natural pause -- a candidate clip boundary.
PAUSE_GAP_THRESHOLD = 0.35

# Generic "hook" words that tend to correlate with attention-grabbing lines.
# This is the zero-key default vocabulary; once a content-analyst run has
# produced real Learnings.top_keywords, those are merged in and take
# priority for scoring going forward.
DEFAULT_HOOK_WORDS = {
    "secret", "mistake", "never", "always", "biggest", "truth", "nobody",
    "everyone", "stop", "why", "how", "worst", "best", "actually", "reason",
    "wrong", "right", "important", "free", "proven", "surprising", "shocking",
    "warning", "avoid", "instead", "real", "honestly", "literally", "huge",
    "massive", "critical", "key", "lesson", "learned", "changed", "genuinely",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "for", "with", "that", "this", "it", "as", "at", "by",
    "be", "have", "has", "had", "i", "you", "we", "they", "he", "she",
    "so", "just", "like", "know", "think", "get", "got", "going", "can",
    "will", "would", "could", "there", "here", "what", "which", "who",
    "im", "its", "not", "do", "does", "did", "about", "if", "then", "back",
}


@dataclass
class _Candidate:
    start: float
    end: float
    text: str
    segment_ids: list[int]


def select_clips(
    transcript: Transcript,
    max_clips: int = DEFAULT_MAX_CLIPS,
    min_len: float = DEFAULT_MIN_LEN,
    max_len: float = DEFAULT_MAX_LEN,
    learnings: Learnings | None = None,
) -> list[Clip]:
    """Zero-key heuristic by default; delegates to the LLM path if a key is
    configured. Both paths return the same ``list[Clip]`` contract."""
    client = get_llm_client()
    if client is None:
        return _select_clips_heuristic(transcript, max_clips, min_len, max_len, learnings)
    return _select_clips_llm(client, transcript, max_clips, min_len, max_len)


def _select_clips_heuristic(
    transcript: Transcript,
    max_clips: int,
    min_len: float,
    max_len: float,
    learnings: Learnings | None,
) -> list[Clip]:
    words = transcript.all_words()
    if not words:
        return []

    boundaries = _find_boundaries(words)
    candidates = _build_candidates(transcript, boundaries, min_len, max_len)
    if not candidates:
        return []

    hook_words = set(DEFAULT_HOOK_WORDS)
    target_len = (min_len + max_len) / 2
    if learnings:
        if learnings.top_keywords:
            hook_words |= {kw.lower() for kw in learnings.top_keywords}
        if learnings.ideal_clip_length_range:
            lo, hi = learnings.ideal_clip_length_range
            target_len = (lo + hi) / 2

    scored = [(_score_candidate(c, hook_words, target_len), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[tuple[float, _Candidate]] = []
    for score, cand in scored:
        if len(selected) >= max_clips:
            break
        if any(_overlaps(cand, s) for _, s in selected):
            continue
        selected.append((score, cand))

    # Output in chronological order -- what a human scanning a clip batch expects.
    selected.sort(key=lambda pair: pair[1].start)

    clips: list[Clip] = []
    for i, (score, cand) in enumerate(selected):
        clips.append(
            Clip(
                id=f"clip-{i + 1:02d}",
                start=round(cand.start, 2),
                end=round(cand.end, 2),
                hook=_extract_hook(cand.text),
                topic=_extract_topic(cand.text),
                score=round(score, 4),
                caption_hint=None,
                source_segment_ids=cand.segment_ids,
                selection_method="heuristic",
            )
        )
    return clips


def _find_boundaries(words: list[Word]) -> list[float]:
    """Timestamps where a pause or sentence-ending punctuation suggests a
    natural clip start/end point."""
    boundaries = {words[0].start, words[-1].end}
    for prev, cur in zip(words, words[1:]):
        gap = cur.start - prev.end
        ends_sentence = prev.text.strip().endswith((".", "!", "?"))
        if gap >= PAUSE_GAP_THRESHOLD or ends_sentence:
            boundaries.add(prev.end)
            boundaries.add(cur.start)
    return sorted(boundaries)


def _build_candidates(
    transcript: Transcript, boundaries: list[float], min_len: float, max_len: float
) -> list[_Candidate]:
    words = transcript.all_words()
    candidates: list[_Candidate] = []
    n = len(boundaries)
    for i in range(n):
        start = boundaries[i]
        for j in range(i + 1, n):
            end = boundaries[j]
            length = end - start
            if length < min_len:
                continue
            if length > max_len:
                break
            window_words = [w for w in words if w.start >= start and w.end <= end]
            if not window_words:
                continue
            text = " ".join(w.text for w in window_words).strip()
            if not text:
                continue
            seg_ids = [seg.id for seg in transcript.segments if seg.end > start and seg.start < end]
            candidates.append(_Candidate(start=start, end=end, text=text, segment_ids=seg_ids))
    return candidates


def _score_candidate(cand: _Candidate, hook_words: set[str], target_len: float) -> float:
    tokens = re.findall(r"[a-zA-Z']+", cand.text.lower())
    if not tokens:
        return 0.0

    hook_hits = sum(1 for t in tokens if t in hook_words)
    keyword_density = hook_hits / len(tokens)

    length = cand.end - cand.start
    length_penalty = ((length - target_len) / target_len) ** 2
    length_score = max(0.0, 1.0 - length_penalty)

    # A clip that opens on a hook word reads better than one where the hook
    # is buried mid-clip.
    opening_bonus = 0.15 if any(t in hook_words for t in tokens[:5]) else 0.0

    return (0.55 * keyword_density * 10) + (0.35 * length_score) + opening_bonus


def _overlaps(a: _Candidate, b: _Candidate) -> bool:
    return a.start < b.end and b.start < a.end


def _extract_hook(text: str, max_words: int = 12) -> str:
    words = text.split()
    hook = " ".join(words[:max_words])
    return hook if hook.endswith((".", "!", "?")) else hook + "..."


def _extract_topic(text: str, max_words: int = 4) -> str:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    counts: dict[str, int] = {}
    for t in tokens:
        if t in STOPWORDS or len(t) < 3:
            continue
        counts[t] = counts.get(t, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:max_words]
    return " ".join(word for word, _ in top) if top else "general"


def _select_clips_llm(
    client: LLMClient, transcript: Transcript, max_clips: int, min_len: float, max_len: float
) -> list[Clip]:
    """LLM path: hand the model the timestamped transcript, force structured
    JSON output matching the Clip contract (see skills/clip-selector/SKILL.md)."""
    system = (
        "You select the best short-form clips from a timestamped transcript for "
        "social media. Pick clips with a strong hook, a complete thought, and a "
        f"duration between {min_len} and {max_len} seconds. Return at most "
        f"{max_clips} clips, non-overlapping, ordered by start time."
    )
    transcript_lines = "\n".join(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}" for seg in transcript.segments)
    schema_hint = (
        '{"clips": [{"start": float, "end": float, "hook": str, "topic": str, '
        '"score": float (0-1), "caption_hint": str}]}'
    )
    result = client.complete_json(system=system, user=transcript_lines, schema_hint=schema_hint)

    clips: list[Clip] = []
    for i, raw in enumerate(result.get("clips", [])[:max_clips]):
        start, end = float(raw["start"]), float(raw["end"])
        clips.append(
            Clip(
                id=f"clip-{i + 1:02d}",
                start=start,
                end=end,
                hook=raw.get("hook", ""),
                topic=raw.get("topic", "general"),
                score=float(raw.get("score", 0.5)),
                caption_hint=raw.get("caption_hint"),
                source_segment_ids=[seg.id for seg in transcript.segments if seg.end > start and seg.start < end],
                selection_method="llm",
            )
        )
    return clips
