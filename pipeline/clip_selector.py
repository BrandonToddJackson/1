"""Transcript -> ranked list[Clip].

Zero-key default is a heuristic scorer: candidate windows are snapped to
pause/sentence boundaries in the word-level timestamps, scored on hook-word
density and closeness to an ideal length, then de-overlapped (non-max
suppression) down to ``max_clips``. If ``pipeline.llm.get_llm_client()``
returns a client, the LLM path is tried instead -- and falls back to the
heuristic path (with a logged warning) if the LLM call fails, returns
unparseable output, or yields zero usable clips after validation, so a bad
LLM response degrades gracefully rather than crashing the run. Same output
contract either way (mirrors skills/clip-selector/SKILL.md).
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass

from pipeline.llm import LLMClient, LLMResponseError, get_llm_client
from pipeline.schemas import Clip, Learnings, Transcript, Word

log = logging.getLogger(__name__)

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
    """A candidate clip window, deliberately lightweight: no text/segment_ids
    stored per-instance (that was the real memory cost on long transcripts --
    tens of thousands of candidates each holding a copy of their own text).
    ``word_lo``/``word_hi`` index into the shared word list; text and segment
    ids are computed on demand by the free functions below."""

    start: float
    end: float
    word_lo: int
    word_hi: int


def _candidate_text(cand: _Candidate, words: list[Word]) -> str:
    return " ".join(w.text for w in words[cand.word_lo : cand.word_hi]).strip()


def _candidate_segment_ids(start: float, end: float, segments: list, seg_starts: list[float], seg_ends: list[float]) -> list[int]:
    """Segment ids overlapping [start, end), found in O(log n) instead of a
    full linear scan per candidate. Only called for the small number of
    finally-selected clips, not for every candidate scored."""
    lo = bisect.bisect_right(seg_ends, start)
    hi = bisect.bisect_left(seg_starts, end)
    hi = max(hi, lo)
    return [segments[k].id for k in range(lo, hi)]


def select_clips(
    transcript: Transcript,
    max_clips: int = DEFAULT_MAX_CLIPS,
    min_len: float = DEFAULT_MIN_LEN,
    max_len: float = DEFAULT_MAX_LEN,
    learnings: Learnings | None = None,
) -> list[Clip]:
    """Zero-key heuristic by default; tries the LLM path first if a key is
    configured, falling back to heuristic on any failure or empty result."""
    client = get_llm_client()
    if client is not None:
        try:
            clips = _select_clips_llm(client, transcript, max_clips, min_len, max_len, learnings)
        except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure degrades gracefully
            log.warning("LLM clip selection failed (%s); falling back to heuristic selection", exc)
        else:
            if clips:
                return clips
            log.warning("LLM clip selection returned no usable clips; falling back to heuristic selection")
    return _select_clips_heuristic(transcript, max_clips, min_len, max_len, learnings)


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
    target_len = max(target_len, 1.0)  # never divide by zero in _score_candidate

    scored = [(_score_candidate(c, words, hook_words, target_len), c) for c in candidates]
    selected = _dedupe_and_limit(scored, max_clips)

    seg_starts = [seg.start for seg in transcript.segments]
    seg_ends = [seg.end for seg in transcript.segments]

    clips: list[Clip] = []
    for i, (score, cand) in enumerate(selected):
        text = _candidate_text(cand, words)
        clips.append(
            Clip(
                id=f"{transcript.run_id}-clip-{i + 1:02d}",
                start=round(cand.start, 2),
                end=round(cand.end, 2),
                hook=_extract_hook(text),
                topic=_extract_topic(text),
                score=round(score, 4),
                caption_hint=None,
                source_segment_ids=_candidate_segment_ids(cand.start, cand.end, transcript.segments, seg_starts, seg_ends),
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
    """For every valid (start, end) boundary pair within [min_len, max_len],
    builds a candidate referencing the words fully inside it via an index
    range -- found with `bisect` in O(log n), not a full linear scan of every
    word per candidate (the confirmed quadratic hot path this replaces:
    8.9s/61MB on a 10-minute transcript, 27.9s/135MB on 20 minutes)."""
    words = transcript.all_words()
    word_starts = [w.start for w in words]

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
            lo = bisect.bisect_left(word_starts, start)
            hi = bisect.bisect_left(word_starts, end)
            # words[lo:hi] all satisfy start&lt;=w.start&lt;end; trim from the right
            # any word that straddles the end boundary (w.end &gt; end) to match
            # the original "fully inside [start, end]" semantics.
            while hi > lo and words[hi - 1].end > end:
                hi -= 1
            if hi <= lo:
                continue
            candidates.append(_Candidate(start=start, end=end, word_lo=lo, word_hi=hi))
    return candidates


def _score_candidate(cand: _Candidate, words: list[Word], hook_words: set[str], target_len: float) -> float:
    text = _candidate_text(cand, words)
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
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


def _overlaps(a, b) -> bool:
    return a.start < b.end and b.start < a.end


def _dedupe_and_limit(scored: list[tuple[float, object]], max_clips: int) -> list[tuple[float, object]]:
    """Greedy non-max suppression shared by both the heuristic and LLM
    selection paths: highest score wins for any overlapping windows, output
    re-sorted chronologically (what a human scanning a clip batch expects).
    Works on anything with .start/.end -- _Candidate or _LLMCandidate."""
    ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
    selected: list[tuple[float, object]] = []
    for score, cand in ranked:
        if len(selected) >= max_clips:
            break
        if any(_overlaps(cand, s) for _, s in selected):
            continue
        selected.append((score, cand))
    selected.sort(key=lambda pair: pair[1].start)
    return selected


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


@dataclass
class _LLMCandidate:
    start: float
    end: float
    hook: str
    topic: str
    caption_hint: str | None
    score: float


def _select_clips_llm(
    client: LLMClient,
    transcript: Transcript,
    max_clips: int,
    min_len: float,
    max_len: float,
    learnings: Learnings | None,
) -> list[Clip]:
    """LLM path: hand the model the timestamped transcript, force structured
    JSON output matching the Clip contract (see skills/clip-selector/SKILL.md).
    Every entry is validated and clamped -- a hallucinated out-of-range
    timestamp or a malformed entry is dropped with a warning, not trusted."""
    system = (
        "You select the best short-form clips from a timestamped transcript for "
        "social media. Pick clips with a strong hook, a complete thought, and a "
        f"duration between {min_len} and {max_len} seconds. Return at most "
        f"{max_clips} clips, non-overlapping, ordered by start time."
    )
    if learnings:
        hints = []
        if learnings.top_keywords:
            hints.append(f"Known high-performing keywords: {', '.join(learnings.top_keywords)}.")
        if learnings.ideal_clip_length_range:
            lo, hi = learnings.ideal_clip_length_range
            hints.append(f"Known ideal clip length range: {lo:.0f}-{hi:.0f}s -- prefer clips near this length.")
        if hints:
            system = system + " " + " ".join(hints)

    transcript_lines = "\n".join(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}" for seg in transcript.segments)
    schema_hint = (
        '{"clips": [{"start": float, "end": float, "hook": str, "topic": str, '
        '"score": float (0-1), "caption_hint": str}]}'
    )
    result = client.complete_json(system=system, user=transcript_lines, schema_hint=schema_hint)

    raw_clips = result.get("clips")
    if not isinstance(raw_clips, list):
        raise LLMResponseError("LLM clip response missing a 'clips' list")

    duration = transcript.duration if transcript.duration and transcript.duration > 0 else None

    candidates: list[_LLMCandidate] = []
    for raw in raw_clips:
        if not isinstance(raw, dict):
            log.warning("LLM clip entry is not an object, skipping: %r", raw)
            continue
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("LLM clip entry missing/invalid start or end, skipping: %s", exc)
            continue
        if end <= start:
            log.warning("LLM clip entry has end <= start, skipping: %r", raw)
            continue
        if duration is not None:
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
        end = min(end, start + max_len)
        if end - start < min_len:
            log.warning("LLM clip entry shorter than min_len after clamping, skipping: %r", raw)
            continue
        try:
            score = float(raw.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        candidates.append(
            _LLMCandidate(
                start=start,
                end=end,
                hook=str(raw.get("hook", "")),
                topic=str(raw.get("topic", "general")),
                caption_hint=raw.get("caption_hint"),
                score=score,
            )
        )

    if not candidates:
        return []

    scored = [(c.score, c) for c in candidates]
    selected = _dedupe_and_limit(scored, max_clips)

    seg_starts = [seg.start for seg in transcript.segments]
    seg_ends = [seg.end for seg in transcript.segments]

    clips: list[Clip] = []
    for i, (score, cand) in enumerate(selected):
        clips.append(
            Clip(
                id=f"{transcript.run_id}-clip-{i + 1:02d}",
                start=round(cand.start, 2),
                end=round(cand.end, 2),
                hook=cand.hook,
                topic=cand.topic,
                score=round(score, 4),
                caption_hint=cand.caption_hint,
                source_segment_ids=_candidate_segment_ids(cand.start, cand.end, transcript.segments, seg_starts, seg_ends),
                selection_method="llm",
            )
        )
    return clips
