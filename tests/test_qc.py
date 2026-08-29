"""Pure-Python tests for pipeline/qc.py's individual checks, each
monkeypatched at its own subprocess-calling helper so no ffmpeg is needed
here. Real-ffmpeg tests (actual probing/measuring against synthetic media)
live in test_qc_media.py."""

from pathlib import Path

from pipeline import qc
from pipeline.schemas import Clip, GraphicsBeat, GraphicsPlan, Word


def _clip(duration=10.0):
    return Clip(id="clip-01", start=0.0, end=duration, hook="h", topic="t", score=0.5)


# ---------------------------------------------------------------------------
# check_duration
# ---------------------------------------------------------------------------

def test_check_duration_within_tolerance_returns_none(monkeypatch):
    monkeypatch.setattr(qc, "_probe_duration", lambda path: 10.1)
    assert qc.check_duration(_clip(10.0), Path("x.mp4")) is None


def test_check_duration_outside_tolerance_flags(monkeypatch):
    monkeypatch.setattr(qc, "_probe_duration", lambda path: 12.0)
    finding = qc.check_duration(_clip(10.0), Path("x.mp4"))
    assert finding is not None
    assert finding.check == "duration"
    assert finding.severity == "warning"
    assert finding.measured == 12.0
    assert finding.expected == 10.0


def test_check_duration_probe_failure_returns_none(monkeypatch):
    monkeypatch.setattr(qc, "_probe_duration", lambda path: None)
    assert qc.check_duration(_clip(), Path("x.mp4")) is None


# ---------------------------------------------------------------------------
# check_loudness
# ---------------------------------------------------------------------------

def test_check_loudness_within_tolerance_returns_none(monkeypatch):
    monkeypatch.setattr(qc, "_measure_loudness", lambda path: -17.0)
    assert qc.check_loudness(_clip(), Path("x.mp4"), target_lufs=-16.0) is None


def test_check_loudness_outside_tolerance_flags(monkeypatch):
    monkeypatch.setattr(qc, "_measure_loudness", lambda path: -30.0)
    finding = qc.check_loudness(_clip(), Path("x.mp4"), target_lufs=-16.0)
    assert finding is not None
    assert finding.check == "loudness"


def test_check_loudness_measurement_failure_returns_none(monkeypatch):
    monkeypatch.setattr(qc, "_measure_loudness", lambda path: None)
    assert qc.check_loudness(_clip(), Path("x.mp4"), target_lufs=-16.0) is None


# ---------------------------------------------------------------------------
# check_silence
# ---------------------------------------------------------------------------

def test_check_silence_parses_silence_durations(monkeypatch):
    class FakeProc:
        stderr = "[silencedetect] silence_start: 1.0\n[silencedetect] silence_end: 2.5 | silence_duration: 1.5\n"

    monkeypatch.setattr(qc, "run_or_raise", lambda cmd, label: FakeProc())
    findings = qc.check_silence(_clip(), Path("x.mp4"), threshold=0.7)
    assert len(findings) == 1
    assert findings[0].measured == 1.5
    assert findings[0].check == "silence"


def test_check_silence_probe_failure_returns_empty(monkeypatch):
    from pipeline import procutil

    def fake_run_or_raise(cmd, label):
        raise procutil.SubprocessFailedError("boom")

    monkeypatch.setattr(qc, "run_or_raise", fake_run_or_raise)
    assert qc.check_silence(_clip(), Path("x.mp4")) == []


def test_check_silence_no_matches_in_stderr_returns_empty(monkeypatch):
    class FakeProc:
        stderr = "no silence detected here"

    monkeypatch.setattr(qc, "run_or_raise", lambda cmd, label: FakeProc())
    assert qc.check_silence(_clip(), Path("x.mp4")) == []


# ---------------------------------------------------------------------------
# check_caption_cues
# ---------------------------------------------------------------------------

def test_check_caption_cues_flags_long_cue():
    # A single sentence with no punctuation break spanning 6s (> MAX_CUE_SECONDS).
    words = [Word(text=f"w{i}", start=float(i), end=float(i) + 0.5) for i in range(7)]
    findings = qc.check_caption_cues(_clip(), words, max_seconds=4.0)
    assert len(findings) >= 1
    assert findings[0].check == "caption_cue_length"


def test_check_caption_cues_short_cues_pass():
    words = [
        Word(text="hi.", start=0.0, end=0.3),
        Word(text="there.", start=0.5, end=0.8),
    ]
    assert qc.check_caption_cues(_clip(), words, max_seconds=4.0) == []


def test_check_caption_cues_empty_words_returns_empty():
    assert qc.check_caption_cues(_clip(), []) == []


# ---------------------------------------------------------------------------
# check_graphics_safe_area
# ---------------------------------------------------------------------------

def test_check_graphics_safe_area_flags_bottom_anchored():
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="lower-third", variables={}, start=1.0, duration=2.0)],
        method="llm",
    )
    findings = qc.check_graphics_safe_area(_clip(), plan)
    assert len(findings) == 1
    assert findings[0].check == "graphics_safe_area"
    assert findings[0].severity == "info"


def test_check_graphics_safe_area_ignores_non_bottom_anchored():
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="stat-card", variables={}, start=1.0, duration=2.0)],
        method="llm",
    )
    assert qc.check_graphics_safe_area(_clip(), plan) == []


def test_check_graphics_safe_area_no_plan_returns_empty():
    assert qc.check_graphics_safe_area(_clip(), None) == []
    assert qc.check_graphics_safe_area(_clip(), GraphicsPlan(clip_id="clip-01", method="skipped")) == []


# ---------------------------------------------------------------------------
# run_checks_for_clip -- aggregation
# ---------------------------------------------------------------------------

def test_run_checks_for_clip_aggregates_all_checks(monkeypatch):
    monkeypatch.setattr(qc, "_probe_duration", lambda path: 10.0)
    monkeypatch.setattr(qc, "_measure_loudness", lambda path: -16.0)
    monkeypatch.setattr(qc, "check_silence", lambda clip, path, threshold=qc.DEAD_AIR_THRESHOLD: [])

    words = [Word(text="hi.", start=0.0, end=0.3)]
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="lower-third", variables={}, start=1.0, duration=2.0)],
        method="llm",
    )
    findings = qc.run_checks_for_clip(_clip(10.0), Path("x.mp4"), words, plan, target_lufs=-16.0)
    # Only the graphics safe-area check should have anything to say here --
    # duration and loudness are dead-on, silence stubbed empty, cues short.
    assert len(findings) == 1
    assert findings[0].check == "graphics_safe_area"


def test_run_checks_for_clip_tolerates_all_probe_failures(monkeypatch):
    monkeypatch.setattr(qc, "_probe_duration", lambda path: None)
    monkeypatch.setattr(qc, "_measure_loudness", lambda path: None)
    monkeypatch.setattr(qc, "check_silence", lambda clip, path, threshold=qc.DEAD_AIR_THRESHOLD: [])

    findings = qc.run_checks_for_clip(_clip(), Path("x.mp4"), [], None, target_lufs=-16.0)
    assert findings == []
