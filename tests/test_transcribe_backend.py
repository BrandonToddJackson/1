"""Pure-Python tests for pipeline/transcribe.py's diarization backend
seam: get_transcribe_backend() precedence/fallback, the ElevenLabs path
(monkeypatched _get_requests(), no real network), the pyannote overlap
assignment (monkeypatched _run_pyannote_pipeline(), no real pyannote.audio
install needed), and transcribe()'s dispatcher degrading one tier down on
any failure."""

import pytest

from pipeline import transcribe
from pipeline.schemas import Transcript, TranscriptSegment, Word


def _clear_backend_env(monkeypatch):
    for var in ("ELEVENLABS_API_KEY", "HUGGINGFACE_TOKEN", "TRANSCRIBE_BACKEND", "DIARIZATION_BACKEND"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# get_transcribe_backend() precedence
# ---------------------------------------------------------------------------

def test_defaults_to_local_with_no_keys(monkeypatch):
    _clear_backend_env(monkeypatch)
    assert transcribe.get_transcribe_backend() == "local"


def test_elevenlabs_key_alone_selects_elevenlabs(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    assert transcribe.get_transcribe_backend() == "elevenlabs"


def test_hf_token_alone_selects_pyannote(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
    assert transcribe.get_transcribe_backend() == "pyannote"


def test_elevenlabs_preferred_over_pyannote_when_both_set(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
    assert transcribe.get_transcribe_backend() == "elevenlabs"


def test_transcribe_backend_local_forces_local_even_with_elevenlabs_key(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "local")
    assert transcribe.get_transcribe_backend() == "local"


def test_transcribe_backend_elevenlabs_without_key_falls_back(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "elevenlabs")
    assert transcribe.get_transcribe_backend() == "local"


def test_diarization_backend_none_disables_pyannote_even_with_token(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
    monkeypatch.setenv("DIARIZATION_BACKEND", "none")
    assert transcribe.get_transcribe_backend() == "local"


def test_diarization_backend_pyannote_without_token_falls_back(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("DIARIZATION_BACKEND", "pyannote")
    assert transcribe.get_transcribe_backend() == "local"


# ---------------------------------------------------------------------------
# ElevenLabs path -- monkeypatched _get_requests(), no real network
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class _FakeRequestsModule:
    def __init__(self, response_json):
        self._response_json = response_json
        self.calls: list[dict] = []

    def post(self, url, headers=None, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "data": data, "files": files, "timeout": timeout})
        return _FakeResponse(self._response_json)


def _elevenlabs_response():
    return {
        "language_code": "en",
        "words": [
            {"type": "word", "text": "hello", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
            {"type": "spacing", "text": " "},
            {"type": "word", "text": "world.", "start": 0.5, "end": 0.9, "speaker_id": "speaker_0"},
            {"type": "audio_event", "text": "(laughter)", "start": 1.0, "end": 1.5},
            {"type": "word", "text": "hi", "start": 2.0, "end": 2.3, "speaker_id": "speaker_1"},
        ],
    }


def test_transcribe_elevenlabs_parses_words_speakers_and_audio_events(monkeypatch, tmp_path):
    fake = _FakeRequestsModule(_elevenlabs_response())
    monkeypatch.setattr(transcribe, "_get_requests", lambda: fake)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake video bytes")

    result = transcribe._transcribe_elevenlabs(media, "run-1")

    assert result.diarization == "elevenlabs"
    assert result.speakers == ["speaker_0", "speaker_1"]
    words = result.all_words()
    assert [w.text for w in words] == ["hello", "world.", "hi"]
    assert words[0].speaker == "speaker_0"
    assert words[2].speaker == "speaker_1"
    assert len(result.audio_events) == 1
    assert result.audio_events[0].type == "laughter"
    assert fake.calls[0]["headers"]["xi-api-key"] == "fake-key"


def test_transcribe_elevenlabs_segments_split_on_speaker_change():
    words = [
        Word(text="hello", start=0.0, end=0.4, speaker="speaker_0"),
        Word(text="world.", start=0.5, end=0.9, speaker="speaker_0"),
        Word(text="hi", start=2.0, end=2.3, speaker="speaker_1"),
    ]
    segments = transcribe._segment_diarized_words(words)
    assert len(segments) == 2
    assert segments[0].speaker == "speaker_0"
    assert segments[1].speaker == "speaker_1"


def test_transcribe_elevenlabs_missing_requests_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "_get_requests", lambda: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    with pytest.raises(RuntimeError):
        transcribe._transcribe_elevenlabs(tmp_path / "source.mp4", "run-1")


# ---------------------------------------------------------------------------
# pyannote overlap assignment -- monkeypatched _run_pyannote_pipeline()
# ---------------------------------------------------------------------------

def test_majority_speaker_for_span_picks_larger_overlap():
    diar = [(0.0, 1.0, "SPEAKER_A"), (1.0, 3.0, "SPEAKER_B")]
    # Word spans mostly SPEAKER_B's segment (1.8s of 2.0s duration).
    assert transcribe._majority_speaker_for_span(0.8, 2.8, diar) == "SPEAKER_B"


def test_majority_speaker_for_span_no_overlap_returns_none():
    diar = [(5.0, 6.0, "SPEAKER_A")]
    assert transcribe._majority_speaker_for_span(0.0, 1.0, diar) is None


def _transcript_for_overlap_test() -> Transcript:
    words = [Word(text="one", start=0.0, end=0.5), Word(text="two", start=2.0, end=2.5)]
    seg = TranscriptSegment(id=0, start=0.0, end=2.5, text="one two", words=words)
    return Transcript(run_id="r1", source_path="x.mp4", duration=2.5, segments=[seg])


def test_assign_speakers_by_overlap_sets_word_and_segment_speaker():
    t = _transcript_for_overlap_test()
    diar = [(0.0, 1.0, "SPEAKER_A"), (1.5, 3.0, "SPEAKER_B")]
    result = transcribe._assign_speakers_by_overlap(t, diar)

    words = result.all_words()
    assert words[0].speaker == "SPEAKER_A"
    assert words[1].speaker == "SPEAKER_B"
    assert result.diarization == "pyannote"
    assert result.speakers == ["SPEAKER_A", "SPEAKER_B"]


def test_assign_speakers_by_overlap_unmatched_word_defaults_speaker00():
    t = _transcript_for_overlap_test()
    diar = [(10.0, 11.0, "SPEAKER_A")]  # no overlap with either word at all
    result = transcribe._assign_speakers_by_overlap(t, diar)
    assert all(w.speaker == "SPEAKER_00" for w in result.all_words())


def test_diarize_pyannote_requires_hf_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    t = _transcript_for_overlap_test()
    with pytest.raises(RuntimeError):
        transcribe._diarize_pyannote(tmp_path / "source.mp4", t)


def test_diarize_pyannote_uses_run_pyannote_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")
    monkeypatch.setattr(
        transcribe, "_run_pyannote_pipeline", lambda media_path, hf_token: [(0.0, 1.0, "SPEAKER_A"), (1.5, 3.0, "SPEAKER_B")]
    )
    t = _transcript_for_overlap_test()
    result = transcribe._diarize_pyannote(tmp_path / "source.mp4", t)
    assert result.diarization == "pyannote"
    assert result.all_words()[1].speaker == "SPEAKER_B"


# ---------------------------------------------------------------------------
# transcribe() dispatcher -- degrades one tier down on any failure
# ---------------------------------------------------------------------------

def test_dispatcher_uses_local_when_no_keys_configured(monkeypatch, tmp_path):
    _clear_backend_env(monkeypatch)
    monkeypatch.setattr(transcribe, "_transcribe_local", lambda media_path, run_id, model_size=None: _transcript_for_overlap_test())
    result = transcribe.transcribe(tmp_path / "source.mp4", "run-1")
    assert result.diarization == "none"


def test_dispatcher_elevenlabs_failure_falls_back_to_local(monkeypatch, tmp_path):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")

    def boom(media_path, run_id):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(transcribe, "_transcribe_elevenlabs", boom)
    monkeypatch.setattr(transcribe, "_transcribe_local", lambda media_path, run_id, model_size=None: _transcript_for_overlap_test())

    result = transcribe.transcribe(tmp_path / "source.mp4", "run-1")
    assert result.diarization == "none"  # fell all the way back to local, no pyannote token configured either


def test_dispatcher_elevenlabs_failure_falls_back_to_pyannote_when_token_present(monkeypatch, tmp_path):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")

    def boom(media_path, run_id):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(transcribe, "_transcribe_elevenlabs", boom)
    monkeypatch.setattr(transcribe, "_transcribe_local", lambda media_path, run_id, model_size=None: _transcript_for_overlap_test())
    monkeypatch.setattr(transcribe, "_diarize_pyannote", lambda media_path, t: t.model_copy(update={"diarization": "pyannote"}))

    result = transcribe.transcribe(tmp_path / "source.mp4", "run-1")
    assert result.diarization == "pyannote"


def test_dispatcher_pyannote_failure_falls_back_to_local_transcript(monkeypatch, tmp_path):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fake-token")

    def boom(media_path, t):
        raise RuntimeError("pyannote exploded")

    monkeypatch.setattr(transcribe, "_transcribe_local", lambda media_path, run_id, model_size=None: _transcript_for_overlap_test())
    monkeypatch.setattr(transcribe, "_diarize_pyannote", boom)

    result = transcribe.transcribe(tmp_path / "source.mp4", "run-1")
    assert result.diarization == "none"  # the local transcript's own default, diarization overlay failed cleanly
