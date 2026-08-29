"""Deterministic, zero-key QC checks against a run's final clip output --
the `qc` command (a non-stage command, sibling of `analyze`/`preview`, not
part of STAGE_ORDER).

Every check here is objective and reproducible: measured duration vs. the
clip's own expected duration (catches EDL math drift in `cut`), measured
loudness vs. the enhance stage's target (catches an `enhance` regression
or a re-encode that drifted), un-removed silence runs longer than
declutter's own dead-air threshold (declutter didn't do its job, or was
off), caption cues on screen longer than a readable maximum, and graphics
beats whose fixed screen position may collide with burned-in captions'
safe area.

No LLM, no human judgment, nothing "does this look right" -- that stays
exactly what it is, a human-in-the-loop step via `preview` (see
pipeline/preview.py and README.md's "the subjective loop" section). QC
answers "is this technically correct", not "is this good."
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.audio import _extract_json_stats
from pipeline.captioner import MAX_WORDS_PER_CUE, _group_into_cues
from pipeline.declutter import DEAD_AIR_THRESHOLD
from pipeline.procutil import SubprocessFailedError, run_or_raise
from pipeline.schemas import Clip, GraphicsPlan, QCFinding, Word

DURATION_TOLERANCE_S = 0.5
LOUDNESS_TOLERANCE_LUFS = 4.0
MAX_CUE_SECONDS = 4.0
SILENCE_NOISE_THRESHOLD_DB = "-30dB"

# Compositions anchored bottom-center/bottom-left -- the same screen region
# every caption style (plain/karaoke/pop) burns in to. Not exact pixel
# geometry (this repo doesn't track per-frame caption bounding boxes), but
# an honest, cheap, always-true-when-flagged heuristic: these two
# compositions WILL share screen space with captions for their whole
# duration, every time.
BOTTOM_ANCHORED_COMPOSITIONS = {"lower-third", "progress-dots"}

_SILENCE_DURATION_RE = re.compile(r"silence_duration:\s*([\d.]+)")


def _probe_duration(path: Path) -> float | None:
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    try:
        proc = run_or_raise(cmd, "qc duration probe")
    except SubprocessFailedError:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def check_duration(clip: Clip, final_path: Path) -> QCFinding | None:
    measured = _probe_duration(final_path)
    if measured is None:
        return None
    expected = clip.duration
    if abs(measured - expected) > DURATION_TOLERANCE_S:
        return QCFinding(
            clip_id=clip.id,
            check="duration",
            severity="warning",
            message=(
                f"measured duration {measured:.2f}s differs from the expected "
                f"{expected:.2f}s by more than {DURATION_TOLERANCE_S}s"
            ),
            measured=measured,
            expected=expected,
        )
    return None


def _measure_loudness(path: Path) -> float | None:
    cmd = ["ffmpeg", "-y", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"]
    try:
        proc = run_or_raise(cmd, "qc loudness measurement")
    except SubprocessFailedError:
        return None
    stats = _extract_json_stats(proc.stderr or "")
    if stats is None:
        return None
    try:
        return float(stats["input_i"])
    except (KeyError, TypeError, ValueError):
        return None


def check_loudness(clip: Clip, final_path: Path, target_lufs: float) -> QCFinding | None:
    measured = _measure_loudness(final_path)
    if measured is None:
        return None
    if abs(measured - target_lufs) > LOUDNESS_TOLERANCE_LUFS:
        return QCFinding(
            clip_id=clip.id,
            check="loudness",
            severity="warning",
            message=(
                f"measured loudness {measured:.1f} LUFS differs from the "
                f"{target_lufs:.1f} LUFS target by more than {LOUDNESS_TOLERANCE_LUFS} LUFS"
            ),
            measured=measured,
            expected=target_lufs,
        )
    return None


def check_silence(clip: Clip, final_path: Path, threshold: float = DEAD_AIR_THRESHOLD) -> list[QCFinding]:
    cmd = ["ffmpeg", "-i", str(final_path), "-af", f"silencedetect=noise={SILENCE_NOISE_THRESHOLD_DB}:d={threshold}", "-f", "null", "-"]
    try:
        proc = run_or_raise(cmd, "qc silence detection")
    except SubprocessFailedError:
        return []
    durations = [float(m) for m in _SILENCE_DURATION_RE.findall(proc.stderr or "")]
    return [
        QCFinding(
            clip_id=clip.id,
            check="silence",
            severity="warning",
            message=f"{d:.2f}s of silence survived declutter's dead-air threshold ({threshold:.2f}s)",
            measured=d,
            expected=threshold,
        )
        for d in durations
    ]


def check_caption_cues(clip: Clip, words: list[Word], max_seconds: float = MAX_CUE_SECONDS) -> list[QCFinding]:
    if not words:
        return []
    cues = _group_into_cues(words, MAX_WORDS_PER_CUE)
    findings = []
    for cue in cues:
        span = cue[-1].end - cue[0].start
        if span > max_seconds:
            preview_text = " ".join(w.text for w in cue)[:40]
            findings.append(
                QCFinding(
                    clip_id=clip.id,
                    check="caption_cue_length",
                    severity="warning",
                    message=f'caption cue "{preview_text}" is on screen {span:.1f}s (> {max_seconds:.1f}s)',
                    measured=span,
                    expected=max_seconds,
                )
            )
    return findings


def check_graphics_safe_area(clip: Clip, graphics_plan: GraphicsPlan | None) -> list[QCFinding]:
    if not graphics_plan or not graphics_plan.beats:
        return []
    findings = []
    for beat in graphics_plan.beats:
        if beat.composition in BOTTOM_ANCHORED_COMPOSITIONS:
            findings.append(
                QCFinding(
                    clip_id=clip.id,
                    check="graphics_safe_area",
                    severity="info",
                    message=(
                        f"beat '{beat.composition}' at {beat.start:.1f}s is bottom-anchored and may "
                        "overlap burned-in captions (also bottom-center) -- review in `preview`"
                    ),
                )
            )
    return findings


def run_checks_for_clip(
    clip: Clip,
    final_path: Path,
    words: list[Word],
    graphics_plan: GraphicsPlan | None,
    target_lufs: float,
) -> list[QCFinding]:
    """Every individual check tolerates its own probe failure (returns
    None/[] rather than raising) -- a QC run should never crash because one
    ffprobe call on one clip failed."""
    findings: list[QCFinding] = []
    duration_finding = check_duration(clip, final_path)
    if duration_finding:
        findings.append(duration_finding)
    loudness_finding = check_loudness(clip, final_path, target_lufs)
    if loudness_finding:
        findings.append(loudness_finding)
    findings.extend(check_silence(clip, final_path))
    findings.extend(check_caption_cues(clip, words))
    findings.extend(check_graphics_safe_area(clip, graphics_plan))
    return findings
