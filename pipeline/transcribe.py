"""MediaAsset -> Transcript, via local faster-whisper (CPU, no API key).

Word-level timestamps, single speaker (SPEAKER_00) -- no diarization in v1
(pyannote-based diarization needs a HuggingFace token; documented as a
future upgrade in README.md).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.config import get_settings
from pipeline.schemas import Transcript, TranscriptSegment, Word


def transcribe(media_path: Path, run_id: str, model_size: str | None = None) -> Transcript:
    from faster_whisper import WhisperModel  # lazy import: heavy dependency

    settings = get_settings()
    model_size = model_size or settings.whisper_model
    compute_type = "int8" if settings.whisper_device == "cpu" else "float16"

    model = WhisperModel(model_size, device=settings.whisper_device, compute_type=compute_type)
    segments_iter, info = model.transcribe(str(media_path), word_timestamps=True)

    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(segments_iter):
        words = [
            Word(text=w.word.strip(), start=w.start, end=w.end, confidence=w.probability)
            for w in (seg.words or [])
            if w.word.strip()
        ]
        segments.append(
            TranscriptSegment(id=i, start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
        )

    return Transcript(
        run_id=run_id,
        source_path=str(media_path),
        language=info.language or "en",
        duration=info.duration or (segments[-1].end if segments else 0.0),
        segments=segments,
        model=f"faster-whisper-{model_size}",
    )
