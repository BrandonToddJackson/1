"""Pydantic models shared across every pipeline stage.

These are the deterministic JSON contracts that flow between stages -- the
same contracts the 4 skills under skills/ describe for their LLM-enabled
counterparts. Keeping them in one module means every stage (and every test)
imports the same shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

_EPS = 1e-6

Platform = Literal["linkedin", "x", "threads", "instagram", "shorts", "newsletter"]

DEFAULT_PLATFORMS: tuple[Platform, ...] = (
    "linkedin",
    "x",
    "threads",
    "instagram",
    "shorts",
    "newsletter",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    # Set only when a diarization backend ran (transcribe.py); None means
    # "unknown/not diarized" -- distinct from the segment-level default
    # single-speaker assignment below, which is always populated.
    speaker: Optional[str] = None


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    # Default single-speaker assignment when no diarization backend ran.
    speaker: str = "SPEAKER_00"


class AudioEvent(BaseModel):
    """A non-speech audio event tagged by a diarization-capable transcribe
    backend (e.g. ElevenLabs Scribe's laughter/applause/sigh tags). Used by
    declutter.py to protect a "dead air" span that actually contains
    laughter from being treated as silence to cut."""

    type: str
    start: float
    end: float


class Transcript(BaseModel):
    run_id: str
    source_path: str
    language: str = "en"
    duration: float
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str = "faster-whisper-base"
    # All default to the pre-diarization shape so a transcript.json written
    # before these fields existed still validates unchanged.
    speakers: list[str] = Field(default_factory=lambda: ["SPEAKER_00"])
    diarization: Literal["none", "pyannote", "elevenlabs"] = "none"
    audio_events: list[AudioEvent] = Field(default_factory=list)

    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def all_words(self) -> list[Word]:
        words: list[Word] = []
        for seg in self.segments:
            words.extend(seg.words)
        return words


class Clip(BaseModel):
    id: str
    start: float
    end: float
    hook: str
    topic: str
    score: float
    caption_hint: Optional[str] = None
    source_segment_ids: list[int] = Field(default_factory=list)
    selection_method: Literal["heuristic", "llm"] = "heuristic"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class Post(BaseModel):
    platform: Platform
    clip_id: Optional[str] = None
    text: str
    hashtags: list[str] = Field(default_factory=list)
    cta: Optional[str] = None
    generation_method: Literal["template", "llm"] = "template"


class MediaAsset(BaseModel):
    run_id: str
    source: str  # original URL or local path as given by the user
    local_path: str
    duration: float = 0.0
    title: Optional[str] = None
    # Set by the enhance stage (audio.py) when it actually ran a filter
    # chain; None in identity mode (enhance disabled) or on assets from
    # before this stage existed -- both mean "local_path is untouched".
    enhanced_from: Optional[str] = None
    loudness_lufs: Optional[float] = None


class PipelineRun(BaseModel):
    run_id: str
    source: str
    created_at: datetime = Field(default_factory=_utcnow)
    stages_completed: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    error: Optional[str] = None
    # Tuning params (max_clips, min_len, max_len, platforms, ...) used for this
    # run -- lets the CLI detect "you asked for different settings than last
    # time" on a resume instead of silently reusing stale output. Absent on
    # PipelineRun.json files written before this field existed; defaults to {}.
    params: dict[str, Any] = Field(default_factory=dict)

    def mark_done(self, stage: str) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)

    def undo(self, stage: str) -> None:
        """Removes a stage from stages_completed (used when invalidating a
        stage and everything downstream of it -- see pipeline/cli.py)."""
        if stage in self.stages_completed:
            self.stages_completed.remove(stage)

    def mark_failed(self, stage: str, exc: BaseException) -> None:
        self.status = "failed"
        self.error = f"{stage}: {exc}"


class PerformanceRecord(BaseModel):
    post_id: str
    platform: str
    clip_id: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    engagement_rate: Optional[float] = None

    def compute_engagement_rate(self) -> float:
        if self.engagement_rate is not None:
            return self.engagement_rate
        if self.views <= 0:
            return 0.0
        return round((self.likes + self.comments + self.shares) / self.views, 4)


class Learnings(BaseModel):
    generated_at: datetime = Field(default_factory=_utcnow)
    top_keywords: list[str] = Field(default_factory=list)
    top_hook_patterns: list[str] = Field(default_factory=list)
    best_platforms: list[str] = Field(default_factory=list)
    ideal_clip_length_range: Optional[tuple[float, float]] = None
    notes: str = ""


class PublishResult(BaseModel):
    platform: str
    method: Literal["outbox", "ayrshare", "blotato"]
    location: str  # local path, or remote post id/url
    status: Literal["ready", "published", "failed"] = "ready"


EditAction = Literal["keep", "remove"]
RemovalReason = Literal["filler", "dead_air", "retake", "false_start", "repetition", "tangent", "manual"]


class EditDecision(BaseModel):
    start: float
    end: float
    action: EditAction
    reason: Optional[RemovalReason] = None  # None for "keep"
    text: str = ""  # words spanned, for human review
    confidence: float = 1.0


class EditPlan(BaseModel):
    """A declutter.py output: the whole source's timeline as an ordered,
    contiguous sequence of keep/remove decisions -- an EDL. pipeline/
    timeline.py's mapping functions assume the invariant this model
    validates: decisions are sorted, non-overlapping, and span exactly
    [0, source_duration). "Disabled" is represented as a single "keep"
    decision spanning the whole source (method="identity"), never as an
    empty/missing plan -- see cli.py's identity-artifact convention."""

    run_id: str
    source_duration: float
    decisions: list[EditDecision] = Field(default_factory=list)
    method: Literal["identity", "heuristic", "llm"] = "identity"
    level: Literal["off", "light", "standard", "aggressive"] = "off"

    @model_validator(mode="after")
    def _validate_contiguous(self) -> "EditPlan":
        if not self.decisions:
            if self.source_duration > _EPS:
                raise ValueError("EditPlan has no decisions but source_duration > 0")
            return self
        ordered = sorted(self.decisions, key=lambda d: d.start)
        if abs(ordered[0].start) > _EPS:
            raise ValueError(f"EditPlan decisions must start at 0, got {ordered[0].start}")
        if abs(ordered[-1].end - self.source_duration) > _EPS:
            raise ValueError(
                f"EditPlan decisions must end at source_duration ({self.source_duration}), got {ordered[-1].end}"
            )
        for a, b in zip(ordered, ordered[1:]):
            if abs(a.end - b.start) > _EPS:
                raise ValueError(f"EditPlan decisions must be contiguous: gap/overlap between {a.end} and {b.start}")
        for d in ordered:
            if d.end < d.start - _EPS:
                raise ValueError(f"EditDecision end ({d.end}) is before start ({d.start})")
        self.decisions = ordered
        return self

    def keep_ranges(self) -> list[tuple[float, float]]:
        return [(d.start, d.end) for d in self.decisions if d.action == "keep"]

    @property
    def clean_duration(self) -> float:
        return round(sum(d.end - d.start for d in self.decisions if d.action == "keep"), 3)

    @property
    def removed_seconds(self) -> float:
        return round(sum(d.end - d.start for d in self.decisions if d.action == "remove"), 3)


class GraphicsBeat(BaseModel):
    """One motion-graphics card, drawn from graphics/catalog.json's fixed
    composition list. `start`/`duration` are clip-relative seconds,
    resolved by graphics.py snapping `anchor_word` to the nearest matching
    transcript Word -- the LLM never places timestamps directly."""

    composition: str
    variables: dict[str, str] = Field(default_factory=dict)
    anchor_word: str = ""
    start: float = 0.0
    duration: float = 0.0  # filled from the catalog entry, never LLM-supplied
    reason: str = ""


class GraphicsPlan(BaseModel):
    clip_id: str
    beats: list[GraphicsBeat] = Field(default_factory=list)
    method: Literal["skipped", "llm"] = "skipped"
    skipped_reason: Optional[str] = None


class QCFinding(BaseModel):
    """One objective, deterministic QC check result -- see pipeline/qc.py.
    Never anything subjective ("does this look good"): that loop stays
    human-in-the-loop via `preview`, documented in README.md."""

    clip_id: str
    check: Literal["duration", "loudness", "silence", "caption_cue_length", "graphics_safe_area"]
    severity: Literal["info", "warning"] = "warning"
    message: str
    measured: Optional[float] = None
    expected: Optional[float] = None


class QCReport(BaseModel):
    run_id: str
    findings: list[QCFinding] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "warning" for f in self.findings)
