"""Transcript -> EditPlan: filler-word and dead-air removal (the `declutter`
stage).

Zero-key default is a heuristic: HARD_FILLERS (um/uh/erm...) are always
removed; SOFT_FILLERS/SOFT_PHRASES (like/basically/you-know...) are removed
only when flanked by a real pause -- evidence they're a verbal tic in that
instance, not doing sentence work ("I like it" keeps "like"; "so, like...
it was good" can lose it). Any inter-word gap longer than DEAD_AIR_THRESHOLD
is trimmed down to a PAUSE_KEEP residual breath, so cuts don't sound like a
hard splice. If pipeline.llm.get_llm_client() returns a client, an LLM pass
additionally proposes retake/false-start/repetition removals, unioned with
(never replacing) the heuristic ones -- and the whole LLM plan is discarded
in favor of heuristic-only if it would remove more than
LLM_MAX_REMOVAL_FRACTION of total duration, a defensive guard against a
model that inverts keep/remove semantics.

Runs at the transcript level, before select_clips (not per-clip after
selection) -- see pipeline/timeline.py, which maps the resulting EditPlan
between the source and clean timelines.

Deliberately does NOT reuse clip_selector.STOPWORDS/PAUSE_GAP_THRESHOLD/
DEFAULT_HOOK_WORDS. Those serve an opposite purpose: topic-labeling wants
"when"/"really" excluded from hashtags, and hook detection WANTS "actually"/
"honestly" flagged as attention-grabbing -- the same words declutter must
treat as disposable filler here. See test_declutter.py's disjointness
assertion (HARD_FILLERS only -- SOFT_FILLERS deliberately overlaps
DEFAULT_HOOK_WORDS, that's the documented divergence, not a bug).

Never merges a removal across a speaker change (today a no-op -- every
Word/segment defaults to a single SPEAKER_00 -- but means diarization
improves declutter later with zero interface change).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pipeline.llm import LLMClient, LLMResponseError, get_llm_client
from pipeline.schemas import AudioEvent, EditDecision, EditPlan, Transcript, Word

log = logging.getLogger(__name__)

DeclutterLevel = Literal["off", "light", "standard", "aggressive"]

# Always removed at any non-off level -- non-lexical disfluencies that never
# carry sentence meaning. Deliberately disjoint from
# clip_selector.DEFAULT_HOOK_WORDS (asserted in tests) -- unlike
# SOFT_FILLERS below, nothing here is ever hook-worthy.
HARD_FILLERS = {"um", "uh", "umm", "uhh", "erm", "mm", "hmm"}

# Removed only when flanked by a pause >= SOFT_FILLER_PAUSE (standard) or
# unconditionally (aggressive). Deliberately overlaps
# clip_selector.DEFAULT_HOOK_WORDS ("actually", "honestly") -- the same word
# is hook-worthy when it opens a sentence with intent and filler when it's
# an isolated tic surrounded by silence; that's what the flanking check is for.
SOFT_FILLERS = {"like", "basically", "literally", "actually", "honestly"}
SOFT_PHRASES: tuple[str, ...] = ("you know", "i mean", "sort of", "kind of")

SOFT_FILLER_PAUSE = 0.2  # seconds of flanking silence required to treat a soft filler as a tic

# An inter-word gap this long is "dead air" -- safe to trim down to
# PAUSE_KEEP. Deliberately much higher than clip_selector.PAUSE_GAP_THRESHOLD
# (0.35s): that threshold answers "where can a clip start" (a low bar); this
# one answers "what silence is safe to delete" (a much higher bar -- a
# thinking pause mid-sentence should survive even though it could also be a
# clip boundary).
DEAD_AIR_THRESHOLD = 0.70
PAUSE_KEEP = 0.25  # residual breath left at every dead-air trim
MIN_REMOVAL = 0.08  # ignore removals shorter than this -- not worth an edit point

LLM_MAX_REMOVAL_FRACTION = 0.40

_Removal = tuple[float, float, str, str]  # (start, end, reason, text)


@dataclass
class _WordCtx:
    word: Word
    speaker: str


def _flatten(transcript: Transcript) -> list[_WordCtx]:
    ctxs: list[_WordCtx] = []
    for seg in transcript.segments:
        for w in seg.words:
            ctxs.append(_WordCtx(word=w, speaker=w.speaker or seg.speaker))
    return ctxs


def _clean_token(text: str) -> str:
    return text.strip(" .,!?;:\"'()-…").lower()


def _is_flanked_by_pause(ctxs: list[_WordCtx], lo: int, hi: int) -> bool:
    """True if the word span ctxs[lo:hi+1] has a pause >= SOFT_FILLER_PAUSE
    on at least one side (start/end of the same-speaker run counts as an
    infinite pause on that side)."""
    if lo == 0 or ctxs[lo - 1].speaker != ctxs[lo].speaker:
        gap_before = float("inf")
    else:
        gap_before = ctxs[lo].word.start - ctxs[lo - 1].word.end
    if hi == len(ctxs) - 1 or ctxs[hi + 1].speaker != ctxs[hi].speaker:
        gap_after = float("inf")
    else:
        gap_after = ctxs[hi + 1].word.start - ctxs[hi].word.end
    return gap_before >= SOFT_FILLER_PAUSE or gap_after >= SOFT_FILLER_PAUSE


def _filler_word_removals(ctxs: list[_WordCtx], include_soft: bool, aggressive: bool) -> list[_Removal]:
    removals: list[_Removal] = []
    for i, ctx in enumerate(ctxs):
        token = _clean_token(ctx.word.text)
        if not token:
            continue
        if token in HARD_FILLERS:
            removals.append((ctx.word.start, ctx.word.end, "filler", ctx.word.text))
        elif include_soft and token in SOFT_FILLERS:
            if aggressive or _is_flanked_by_pause(ctxs, i, i):
                removals.append((ctx.word.start, ctx.word.end, "filler", ctx.word.text))
    return removals


def _soft_phrase_removals(ctxs: list[_WordCtx], aggressive: bool) -> list[_Removal]:
    removals: list[_Removal] = []
    tokens = [_clean_token(c.word.text) for c in ctxs]
    n = len(ctxs)
    for phrase in SOFT_PHRASES:
        phrase_tokens = tuple(phrase.split())
        plen = len(phrase_tokens)
        i = 0
        while i <= n - plen:
            span = tuple(tokens[i : i + plen])
            hi = i + plen - 1
            if span == phrase_tokens and ctxs[i].speaker == ctxs[hi].speaker:
                if aggressive or _is_flanked_by_pause(ctxs, i, hi):
                    start = ctxs[i].word.start
                    end = ctxs[hi].word.end
                    text = " ".join(c.word.text for c in ctxs[i : hi + 1])
                    removals.append((start, end, "filler", text))
                    i += plen  # don't allow overlapping matches
                    continue
            i += 1
    return removals


def _overlaps_any_event(start: float, end: float, events: list[AudioEvent]) -> bool:
    return any(ev.start < end and start < ev.end for ev in events)


def _dead_air_removals(ctxs: list[_WordCtx], audio_events: list[AudioEvent]) -> list[_Removal]:
    removals: list[_Removal] = []
    for prev, cur in zip(ctxs, ctxs[1:]):
        if prev.speaker != cur.speaker:
            continue  # never cut across a speaker change
        gap = cur.word.start - prev.word.end
        if gap < DEAD_AIR_THRESHOLD:
            continue
        trim_start = prev.word.end + PAUSE_KEEP / 2
        trim_end = cur.word.start - PAUSE_KEEP / 2
        if trim_end - trim_start < MIN_REMOVAL:
            continue
        if _overlaps_any_event(trim_start, trim_end, audio_events):
            continue  # e.g. laughter/applause tagged by a diarization backend -- not silence
        removals.append((trim_start, trim_end, "dead_air", ""))
    return removals


def _heuristic_removals(transcript: Transcript, level: DeclutterLevel) -> list[_Removal]:
    ctxs = _flatten(transcript)
    if not ctxs:
        return []
    include_soft = level in ("standard", "aggressive")
    aggressive = level == "aggressive"

    removals = _filler_word_removals(ctxs, include_soft, aggressive)
    if include_soft:
        removals.extend(_soft_phrase_removals(ctxs, aggressive))
    removals.extend(_dead_air_removals(ctxs, transcript.audio_events))
    return removals


def _merge_spans(spans: list[_Removal]) -> list[_Removal]:
    """Sorted, overlapping/touching spans merged; earliest span's reason
    wins, text is concatenated (deduped) across whatever merged into it."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s[0])
    merged: list[list] = [list(ordered[0])]
    for start, end, reason, text in ordered[1:]:
        last = merged[-1]
        if start <= last[1] + 1e-9:
            last[1] = max(last[1], end)
            if text and text not in last[3]:
                last[3] = (last[3] + " " + text).strip()
        else:
            merged.append([start, end, reason, text])
    return [(s, e, r, t) for s, e, r, t in merged]


def _identity_plan(run_id: str, duration: float) -> EditPlan:
    if duration <= 0:
        return EditPlan(run_id=run_id, source_duration=0.0, decisions=[], method="identity", level="off")
    return EditPlan(
        run_id=run_id,
        source_duration=duration,
        decisions=[EditDecision(start=0.0, end=duration, action="keep")],
        method="identity",
        level="off",
    )


def _build_plan(run_id: str, duration: float, removals: list[_Removal], method: str, level: DeclutterLevel) -> EditPlan:
    decisions: list[EditDecision] = []
    cursor = 0.0
    for start, end, reason, text in sorted(removals, key=lambda r: r[0]):
        start = max(start, cursor)
        end = min(end, duration)
        if start >= end:
            continue
        if start > cursor:
            decisions.append(EditDecision(start=cursor, end=start, action="keep"))
        decisions.append(EditDecision(start=start, end=end, action="remove", reason=reason, text=text))
        cursor = end
    if cursor < duration:
        decisions.append(EditDecision(start=cursor, end=duration, action="keep"))
    if not decisions:
        decisions = [EditDecision(start=0.0, end=duration, action="keep")]
    return EditPlan(run_id=run_id, source_duration=duration, decisions=decisions, method=method, level=level)


def _llm_removals(client: LLMClient, transcript: Transcript) -> list[_Removal]:
    system = (
        "You find retakes, false starts, and verbatim repetitions in a timestamped "
        "transcript of someone recording themselves talking to camera. Return ONLY "
        "spans that should be CUT (removed) -- never spans to keep. A retake is when "
        "the speaker restarts a sentence or thought; keep only the final, best "
        "version and mark the earlier attempt(s) for removal. Timestamps must be in "
        "seconds and must fall within the transcript's own timestamps. Be "
        "conservative: when in doubt, don't mark it."
    )
    transcript_lines = "\n".join(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}" for seg in transcript.segments)
    schema_hint = (
        '{"removals": [{"start": float, "end": float, '
        '"reason": "retake"|"false_start"|"repetition", "text": str}]}'
    )
    result = client.complete_json(system=system, user=transcript_lines, schema_hint=schema_hint)

    raw = result.get("removals")
    if not isinstance(raw, list):
        raise LLMResponseError("LLM declutter response missing a 'removals' list")

    duration = transcript.duration
    valid_reasons = {"retake", "false_start", "repetition"}
    removals: list[_Removal] = []
    for entry in raw:
        if not isinstance(entry, dict):
            log.warning("LLM declutter entry is not an object, skipping: %r", entry)
            continue
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("LLM declutter entry missing/invalid start or end, skipping: %s", exc)
            continue
        if end <= start:
            log.warning("LLM declutter entry has end <= start, skipping: %r", entry)
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < MIN_REMOVAL:
            continue
        reason = entry.get("reason")
        if reason not in valid_reasons:
            reason = "retake"
        removals.append((start, end, reason, str(entry.get("text", ""))))
    return removals


def declutter(transcript: Transcript, level: DeclutterLevel = "standard") -> EditPlan:
    """Zero-key heuristic by default (see module docstring for the level
    ladder: light=hard-only, standard=hard+flanked-soft,
    aggressive=hard+all-soft; dead-air trimming runs at every non-off
    level). If an LLM client is configured, its retake/false-start/
    repetition removals are unioned in -- discarded (falls back to
    heuristic-only) if they'd remove more than LLM_MAX_REMOVAL_FRACTION of
    total duration."""
    if level == "off" or transcript.duration <= 0:
        return _identity_plan(transcript.run_id, transcript.duration)

    heuristic = _heuristic_removals(transcript, level)

    llm_removals: list[_Removal] = []
    client = get_llm_client()
    if client is not None:
        try:
            candidate = _llm_removals(client, transcript)
        except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure degrades gracefully
            log.warning("LLM declutter pass failed (%s); using heuristic-only plan", exc)
        else:
            total = sum(e - s for s, e, _, _ in candidate)
            if transcript.duration > 0 and total / transcript.duration > LLM_MAX_REMOVAL_FRACTION:
                log.warning(
                    "LLM declutter plan would remove %.0f%% of duration (> %.0f%% guard); "
                    "discarding LLM removals, using heuristic-only plan",
                    100 * total / transcript.duration, 100 * LLM_MAX_REMOVAL_FRACTION,
                )
            else:
                llm_removals = candidate

    merged = [(s, e, r, t) for s, e, r, t in _merge_spans(heuristic + llm_removals) if e - s >= MIN_REMOVAL]
    method = "llm" if llm_removals else "heuristic"
    return _build_plan(transcript.run_id, transcript.duration, merged, method=method, level=level)
