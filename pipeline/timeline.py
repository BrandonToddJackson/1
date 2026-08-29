"""Source-timeline <-> clean-timeline mapping, driven by an EditPlan.

declutter.py produces an EditPlan describing which spans of the ORIGINAL
source survive (the "clean" timeline the viewer actually experiences).
Every stage downstream of declutter (select_clips, cut, caption, repurpose)
reasons in clean-timeline coordinates -- these are the only functions that
know how to translate between the two. Pure Python, no I/O, so this is the
cheap, exhaustively-testable part of the whole feature.
"""

from __future__ import annotations

import bisect

from pipeline.schemas import EditPlan, Transcript, TranscriptSegment, Word

_EPS = 1e-6


def _keep_ranges_and_offsets(plan: EditPlan) -> tuple[list[tuple[float, float]], list[float]]:
    """(keep ranges in source time, cumulative clean-time offset at the
    START of each keep range) -- parallel arrays, both already sorted since
    EditPlan's validator guarantees decisions are ordered."""
    ranges = plan.keep_ranges()
    offsets: list[float] = []
    cum = 0.0
    for start, end in ranges:
        offsets.append(cum)
        cum += end - start
    return ranges, offsets


def clean_to_source(plan: EditPlan, t_clean: float) -> float:
    """Maps a clean-timeline timestamp back to the original source
    timestamp. Raises ValueError if t_clean is outside [0, clean_duration]
    or the plan has no surviving content."""
    ranges, offsets = _keep_ranges_and_offsets(plan)
    if not ranges:
        raise ValueError("EditPlan has no keep ranges -- nothing survived declutter")
    if t_clean < -_EPS or t_clean > plan.clean_duration + _EPS:
        raise ValueError(f"t_clean={t_clean} outside [0, {plan.clean_duration}]")
    idx = max(0, min(bisect.bisect_right(offsets, t_clean) - 1, len(ranges) - 1))
    start, end = ranges[idx]
    source_time = start + (t_clean - offsets[idx])
    return min(source_time, end)  # clamp float slop at the range's own edge


def source_to_clean(plan: EditPlan, t_source: float) -> float | None:
    """Maps a source timestamp to its clean-timeline equivalent, or None if
    t_source falls inside a removed span (there is no clean-timeline
    equivalent for content that was cut)."""
    ranges, offsets = _keep_ranges_and_offsets(plan)
    if not ranges:
        return None
    starts = [r[0] for r in ranges]
    idx = bisect.bisect_right(starts, t_source) - 1
    if idx < 0:
        return None
    start, end = ranges[idx]
    if t_source < start - _EPS or t_source > end + _EPS:
        return None
    return offsets[idx] + (t_source - start)


def source_ranges_for(plan: EditPlan, clean_start: float, clean_end: float) -> list[tuple[float, float]]:
    """Given a clip's [clean_start, clean_end) on the CLEAN timeline,
    returns the source-timeline (start, end) sub-ranges that must be
    extracted and concatenated (via cutter.cut_ranges) to reconstruct it --
    possibly several, if the clip spans a declutter cut boundary."""
    if clean_end <= clean_start:
        return []
    ranges, offsets = _keep_ranges_and_offsets(plan)
    result: list[tuple[float, float]] = []
    for (r_start, r_end), off in zip(ranges, offsets):
        r_clean_start = off
        r_clean_end = off + (r_end - r_start)
        overlap_start = max(clean_start, r_clean_start)
        overlap_end = min(clean_end, r_clean_end)
        if overlap_end - overlap_start > _EPS:
            src_start = r_start + (overlap_start - r_clean_start)
            src_end = r_start + (overlap_end - r_clean_start)
            result.append((src_start, src_end))
    return result


def merge_small_gaps(ranges: list[tuple[float, float]], max_ranges: int) -> list[tuple[float, float]]:
    """If `ranges` exceeds max_ranges, repeatedly merges the pair of
    adjacent ranges separated by the smallest gap until the count fits --
    keeps a cut's filtergraph size bounded (cutter.MAX_KEEP_RANGES) without
    an arbitrary truncation that would silently drop content."""
    merged = sorted(ranges)
    while len(merged) > max_ranges and len(merged) > 1:
        gaps = [(merged[i + 1][0] - merged[i][1], i) for i in range(len(merged) - 1)]
        gaps.sort(key=lambda pair: pair[0])
        _, i = gaps[0]
        joined = (merged[i][0], merged[i + 1][1])
        merged = merged[:i] + [joined] + merged[i + 2 :]
    return merged


def apply_plan_to_transcript(t: Transcript, plan: EditPlan) -> Transcript:
    """Rebuilds a Transcript onto the clean timeline: drops any word that
    isn't fully inside a surviving span, remaps surviving words' start/end
    via source_to_clean, re-derives each segment's text from its surviving
    words, drops segments left with no words, and renumbers what remains
    0..N-1. select_clips/repurposer/captioner consume the result exactly
    like any other Transcript -- no changes needed downstream."""
    new_segments: list[TranscriptSegment] = []
    next_id = 0
    for seg in t.segments:
        new_words: list[Word] = []
        for w in seg.words:
            clean_start = source_to_clean(plan, w.start)
            clean_end = source_to_clean(plan, w.end)
            if clean_start is None or clean_end is None:
                continue
            new_words.append(w.model_copy(update={"start": clean_start, "end": clean_end}))
        if not new_words:
            continue
        new_segments.append(
            TranscriptSegment(
                id=next_id,
                start=new_words[0].start,
                end=new_words[-1].end,
                text=" ".join(w.text for w in new_words),
                words=new_words,
                speaker=seg.speaker,
            )
        )
        next_id += 1

    return t.model_copy(update={"segments": new_segments, "duration": plan.clean_duration})
