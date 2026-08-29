"""Pydantic models shared across every pipeline stage.

These are the deterministic JSON contracts that flow between stages -- the
same contracts the 4 skills under skills/ describe for their LLM-enabled
counterparts. Keeping them in one module means every stage (and every test)
imports the same shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

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


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)
    # No diarization in v1 -- always the single default speaker.
    speaker: str = "SPEAKER_00"


class Transcript(BaseModel):
    run_id: str
    source_path: str
    language: str = "en"
    duration: float
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str = "faster-whisper-base"

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


class PipelineRun(BaseModel):
    run_id: str
    source: str
    created_at: datetime = Field(default_factory=_utcnow)
    stages_completed: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    error: Optional[str] = None

    def mark_done(self, stage: str) -> None:
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)


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
    method: Literal["outbox", "ayrshare"]
    location: str  # local path, or remote post id/url
    status: Literal["ready", "published", "failed"] = "ready"
