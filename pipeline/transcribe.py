"""MediaAsset -> Transcript, with an optional diarization overlay.

Zero-key default (unchanged): local faster-whisper (CPU), word-level
timestamps, single speaker (SPEAKER_00), `diarization="none"`.

get_transcribe_backend() is the single decision point (mirrors
pipeline.llm.get_llm_client()) resolving which of three tiers actually
runs, in precedence order:

1. **ElevenLabs Scribe** (paid) -- replaces the whole call: one request
   returns transcription + diarization + non-speech audio-event tags
   (laughter, applause, ...) together. UNVERIFIED: reconstructed from
   ElevenLabs' Speech-to-Text documentation, not verified against a live
   key -- api.elevenlabs.io was egress-blocked during implementation (the
   same honest caveat pipeline/publisher.py already carries for Blotato).
   Verify against a real key and
   https://elevenlabs.io/docs/api-reference/speech-to-text before relying
   on this in production.
2. **faster-whisper + pyannote.audio** (local, free-but-gated) -- today's
   local transcript, then a diarization overlay assigning a speaker to
   every word by MAJORITY TEMPORAL OVERLAP (not nearest-boundary, which
   flips speakers on crosstalk) with pyannote's diarization spans, then a
   per-segment majority vote among its own words. Needs a free
   HUGGINGFACE_TOKEN (pyannote's pretrained pipeline is gated behind
   accepting its license on huggingface.co).
3. **faster-whisper alone** (today's exact behavior) -- the default.

Every tier degrades ONE level down on any failure (network error, missing
token, bad response), logged as a warning, never raises -- transcribe()
always returns a usable Transcript.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

from pipeline.config import get_settings
from pipeline.schemas import AudioEvent, Transcript, TranscriptSegment, Word

log = logging.getLogger(__name__)

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_MODEL_ID = "scribe_v1"
PYANNOTE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"

# A new segment starts on a speaker change, a pause this long, or after
# sentence-ending punctuation -- mirrors captioner._group_into_cues's
# boundary logic, but speaker-aware (used only for ElevenLabs' flat word
# list, which arrives with no segment grouping of its own).
_SEGMENT_PAUSE_THRESHOLD = 1.0


def _get_requests() -> ModuleType | None:
    """Lazy, test-friendly import seam -- same pattern as
    publisher.py::_get_requests. `requests` is only needed on the
    ElevenLabs path (the `elevenlabs` extra, not a base dependency)."""
    try:
        import requests
    except ImportError:
        return None
    return requests


def get_transcribe_backend() -> str:
    """Resolves to "elevenlabs" | "pyannote" | "local". Two independent
    overrides (both settings default "auto"):
      TRANSCRIBE_BACKEND=local forces local even if an ElevenLabs key is
        configured; =elevenlabs requires the key (falls back, logged, if
        it's missing).
      DIARIZATION_BACKEND=none disables the pyannote overlay even if a
        HuggingFace token is configured (skips its first-run model
        download without unsetting the token for other uses); =pyannote
        requires the token (falls back, logged, if it's missing).
    """
    settings = get_settings()
    tb = (settings.transcribe_backend or "auto").lower()
    db = (settings.diarization_backend or "auto").lower()

    wants_elevenlabs = tb == "elevenlabs" or (tb == "auto" and bool(settings.elevenlabs_api_key))
    if wants_elevenlabs:
        if settings.elevenlabs_api_key:
            return "elevenlabs"
        log.warning("TRANSCRIBE_BACKEND=elevenlabs but ELEVENLABS_API_KEY is not set; falling back to local transcribe")

    wants_pyannote = db == "pyannote" or (db == "auto" and bool(settings.huggingface_token))
    if wants_pyannote:
        if settings.huggingface_token:
            return "pyannote"
        log.warning("DIARIZATION_BACKEND=pyannote but HUGGINGFACE_TOKEN is not set; using local transcribe with no diarization")

    return "local"


def transcribe(media_path: Path, run_id: str, model_size: str | None = None) -> Transcript:
    backend = get_transcribe_backend()

    if backend == "elevenlabs":
        try:
            return _transcribe_elevenlabs(media_path, run_id)
        except Exception as exc:  # noqa: BLE001 - degrade one tier, never crash the run
            log.warning("ElevenLabs transcribe failed (%s); falling back to local transcribe", exc)
            backend = "pyannote" if get_settings().huggingface_token else "local"

    transcript = _transcribe_local(media_path, run_id, model_size)

    if backend == "pyannote":
        try:
            transcript = _diarize_pyannote(media_path, transcript)
        except Exception as exc:  # noqa: BLE001
            log.warning("pyannote diarization failed (%s); using single-speaker transcript", exc)

    return transcript


def _transcribe_local(media_path: Path, run_id: str, model_size: str | None = None) -> Transcript:
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


# ---------------------------------------------------------------------------
# ElevenLabs Scribe (transcription + diarization + audio events, one call)
# ---------------------------------------------------------------------------

def _segment_diarized_words(words: list[Word]) -> list[TranscriptSegment]:
    """Groups a flat, already-diarized word list into segments. ElevenLabs'
    response is a flat word list with no segment grouping of its own."""
    if not words:
        return []
    segments: list[TranscriptSegment] = []
    current: list[Word] = [words[0]]
    for prev, cur in zip(words, words[1:]):
        speaker_changed = (cur.speaker or "SPEAKER_00") != (prev.speaker or "SPEAKER_00")
        gap = cur.start - prev.end
        sentence_end = prev.text.strip().endswith((".", "!", "?"))
        if speaker_changed or gap >= _SEGMENT_PAUSE_THRESHOLD or sentence_end:
            segments.append(_make_segment(len(segments), current))
            current = []
        current.append(cur)
    if current:
        segments.append(_make_segment(len(segments), current))
    return segments


def _make_segment(seg_id: int, words: list[Word]) -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        words=words,
        speaker=words[0].speaker or "SPEAKER_00",
    )


def _transcribe_elevenlabs(media_path: Path, run_id: str) -> Transcript:
    """See module docstring's UNVERIFIED caveat. Reconstructed request/
    response shape: POST multipart/form-data to ELEVENLABS_STT_URL, header
    `xi-api-key`, fields `model_id`/`diarize`/`tag_audio_events`, response
    `{"language_code": ..., "words": [{"type": "word"|"spacing"|
    "audio_event", "text": ..., "start": ..., "end": ..., "speaker_id": ...}]}`."""
    requests = _get_requests()
    if requests is None:
        raise RuntimeError("requests not installed - pip install -e '.[elevenlabs]'")

    settings = get_settings()
    with open(media_path, "rb") as f:
        resp = requests.post(
            ELEVENLABS_STT_URL,
            headers={"xi-api-key": settings.elevenlabs_api_key},
            data={"model_id": ELEVENLABS_MODEL_ID, "diarize": "true", "tag_audio_events": "true"},
            files={"file": (media_path.name, f)},
            timeout=600,
        )
    resp.raise_for_status()
    data = resp.json()

    words: list[Word] = []
    audio_events: list[AudioEvent] = []
    for item in data.get("words", []):
        item_type = item.get("type")
        if item_type == "word":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(item["start"]), end=float(item["end"]), speaker=item.get("speaker_id")))
        elif item_type == "audio_event":
            event_type = (item.get("text") or "unknown").strip("() ") or "unknown"
            audio_events.append(AudioEvent(type=event_type, start=float(item.get("start", 0.0)), end=float(item.get("end", 0.0))))
        # "spacing" entries carry no useful timing/text -- ignored.

    segments = _segment_diarized_words(words)
    speakers = sorted({w.speaker for w in words if w.speaker}) or ["SPEAKER_00"]
    duration = words[-1].end if words else 0.0

    return Transcript(
        run_id=run_id,
        source_path=str(media_path),
        language=data.get("language_code") or "en",
        duration=duration,
        segments=segments,
        model=f"elevenlabs-{ELEVENLABS_MODEL_ID}",
        speakers=speakers,
        diarization="elevenlabs",
        audio_events=audio_events,
    )


# ---------------------------------------------------------------------------
# pyannote.audio diarization overlay (on top of a local faster-whisper pass)
# ---------------------------------------------------------------------------

def _run_pyannote_pipeline(media_path: Path, hf_token: str) -> list[tuple[float, float, str]]:
    """Returns [(start, end, speaker_label), ...]. Isolated in its own
    function so tests can monkeypatch it instead of needing pyannote.audio
    actually installed -- a large, optional dependency (see pyproject.toml's
    [diarize] extra)."""
    from pyannote.audio import Pipeline  # lazy import: heavy optional dependency

    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE_ID, use_auth_token=hf_token)
    annotation = pipeline(str(media_path))
    return [(segment.start, segment.end, speaker) for segment, _, speaker in annotation.itertracks(yield_label=True)]


def _majority_speaker_for_span(start: float, end: float, diar_segments: list[tuple[float, float, str]]) -> str | None:
    """Majority TEMPORAL OVERLAP, not nearest-boundary -- nearest-boundary
    flips speakers on crosstalk (a word starting a beat before the other
    speaker's diarization segment technically begins would get assigned to
    the wrong speaker; overlap-weighting is robust to that). Returns None
    if the span has no overlap with any diarization segment at all."""
    overlap_by_speaker: dict[str, float] = {}
    for seg_start, seg_end, speaker in diar_segments:
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > 0:
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
    if not overlap_by_speaker:
        return None
    return max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]


def _assign_speakers_by_overlap(transcript: Transcript, diar_segments: list[tuple[float, float, str]]) -> Transcript:
    """Word speakers first (majority overlap against the raw diarization
    spans), then each TranscriptSegment's own speaker is a majority vote
    among ITS OWN already-assigned words -- keeps segment- and word-level
    speaker assignment consistent with each other by construction, rather
    than computing them independently from the raw spans twice."""
    new_segments: list[TranscriptSegment] = []
    for seg in transcript.segments:
        new_words = [
            w.model_copy(update={"speaker": _majority_speaker_for_span(w.start, w.end, diar_segments) or "SPEAKER_00"})
            for w in seg.words
        ]
        if new_words:
            counts: dict[str, int] = {}
            for w in new_words:
                counts[w.speaker] = counts.get(w.speaker, 0) + 1
            seg_speaker = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            seg_speaker = seg.speaker
        new_segments.append(seg.model_copy(update={"words": new_words, "speaker": seg_speaker}))

    speakers = sorted({w.speaker for seg in new_segments for w in seg.words if w.speaker}) or ["SPEAKER_00"]
    return transcript.model_copy(update={"segments": new_segments, "speakers": speakers, "diarization": "pyannote"})


def _diarize_pyannote(media_path: Path, transcript: Transcript) -> Transcript:
    settings = get_settings()
    if not settings.huggingface_token:
        raise RuntimeError("HUGGINGFACE_TOKEN not set")
    diar_segments = _run_pyannote_pipeline(media_path, settings.huggingface_token)
    return _assign_speakers_by_overlap(transcript, diar_segments)
