import pytest

from pipeline import clip_selector, declutter
from pipeline.schemas import AudioEvent, Transcript, TranscriptSegment, Word


def _transcript(words: list[Word], duration: float, audio_events: list[AudioEvent] | None = None) -> Transcript:
    seg = TranscriptSegment(id=0, start=0.0, end=duration, text=" ".join(w.text for w in words), words=words)
    return Transcript(run_id="r1", source_path="x.mp4", duration=duration, segments=[seg], audio_events=audio_events or [])


# ---------------------------------------------------------------------------
# Deliberate vocabulary divergence from clip_selector
# ---------------------------------------------------------------------------

def test_hard_fillers_disjoint_from_hook_words():
    assert declutter.HARD_FILLERS.isdisjoint(clip_selector.DEFAULT_HOOK_WORDS)


# ---------------------------------------------------------------------------
# Identity plan
# ---------------------------------------------------------------------------

def test_level_off_returns_identity_plan():
    words = [Word(text="um", start=0.0, end=0.3), Word(text="hi", start=0.5, end=0.8)]
    t = _transcript(words, duration=0.8)
    plan = declutter.declutter(t, level="off")
    assert plan.method == "identity"
    assert plan.level == "off"
    assert plan.decisions == [plan.decisions[0]]
    assert plan.decisions[0].action == "keep"
    assert plan.removed_seconds == 0.0


def test_zero_duration_returns_identity_plan():
    t = _transcript([], duration=0.0)
    plan = declutter.declutter(t, level="standard")
    assert plan.decisions == []
    assert plan.clean_duration == 0.0


# ---------------------------------------------------------------------------
# Hard fillers -- removed at every non-off level
# ---------------------------------------------------------------------------

def test_hard_filler_removed_at_light_level():
    words = [
        Word(text="so", start=0.0, end=0.3),
        Word(text="um", start=0.4, end=0.6),
        Word(text="let's", start=0.7, end=1.0),
        Word(text="begin", start=1.0, end=1.3),
    ]
    t = _transcript(words, duration=1.3)
    plan = declutter.declutter(t, level="light")

    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1
    assert removed[0].reason == "filler"
    assert removed[0].start == pytest.approx(0.4)
    assert removed[0].end == pytest.approx(0.6)
    assert plan.method == "heuristic"


def test_hard_filler_short_removal_dropped_below_min_removal():
    words = [
        Word(text="um", start=0.0, end=0.03),  # 0.03s < MIN_REMOVAL (0.08)
        Word(text="hi", start=0.5, end=0.8),
    ]
    t = _transcript(words, duration=0.8)
    plan = declutter.declutter(t, level="light")
    assert all(d.action == "keep" for d in plan.decisions)


# ---------------------------------------------------------------------------
# Soft fillers -- level-gated + flanking-pause-gated
# ---------------------------------------------------------------------------

def _soft_filler_transcript(contiguous: bool) -> Transcript:
    if contiguous:
        words = [
            Word(text="I", start=0.0, end=0.2),
            Word(text="like", start=0.2, end=0.4),
            Word(text="it", start=0.4, end=0.6),
        ]
        duration = 0.6
    else:
        words = [
            Word(text="well", start=0.0, end=0.3),
            Word(text="like", start=0.6, end=0.8),  # 0.3s pause before
            Word(text="it", start=1.2, end=1.4),  # 0.4s pause after
        ]
        duration = 1.4
    return _transcript(words, duration=duration)


def test_soft_filler_never_removed_at_light():
    t = _soft_filler_transcript(contiguous=False)
    plan = declutter.declutter(t, level="light")
    assert all(d.action == "keep" for d in plan.decisions)


def test_soft_filler_not_removed_at_standard_when_not_flanked():
    t = _soft_filler_transcript(contiguous=True)
    plan = declutter.declutter(t, level="standard")
    assert all(d.action == "keep" for d in plan.decisions)


def test_soft_filler_removed_at_standard_when_flanked():
    t = _soft_filler_transcript(contiguous=False)
    plan = declutter.declutter(t, level="standard")
    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1
    assert removed[0].start == pytest.approx(0.6)
    assert removed[0].end == pytest.approx(0.8)


def test_soft_filler_removed_at_aggressive_even_unflanked():
    t = _soft_filler_transcript(contiguous=True)
    plan = declutter.declutter(t, level="aggressive")
    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1
    assert removed[0].start == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Soft phrases
# ---------------------------------------------------------------------------

def test_soft_phrase_removed_when_flanked():
    words = [
        Word(text="well", start=0.0, end=0.3),
        Word(text="you", start=0.6, end=0.7),
        Word(text="know", start=0.7, end=0.8),
        Word(text="it's", start=1.1, end=1.3),
    ]
    t = _transcript(words, duration=1.3)
    plan = declutter.declutter(t, level="standard")
    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1
    assert removed[0].start == pytest.approx(0.6)
    assert removed[0].end == pytest.approx(0.8)
    assert removed[0].text == "you know"


# ---------------------------------------------------------------------------
# Dead air
# ---------------------------------------------------------------------------

def test_dead_air_trimmed_leaves_pause_keep_residual():
    words = [Word(text="one", start=0.0, end=1.0), Word(text="two", start=2.0, end=3.0)]
    t = _transcript(words, duration=3.0)
    plan = declutter.declutter(t, level="light")

    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1
    assert removed[0].reason == "dead_air"
    assert removed[0].start == pytest.approx(1.0 + declutter.PAUSE_KEEP / 2)
    assert removed[0].end == pytest.approx(2.0 - declutter.PAUSE_KEEP / 2)
    assert plan.removed_seconds == pytest.approx(1.0 - declutter.PAUSE_KEEP)


def test_dead_air_not_removed_across_speaker_change():
    words = [
        Word(text="one", start=0.0, end=1.0, speaker="SPEAKER_00"),
        Word(text="two", start=2.0, end=3.0, speaker="SPEAKER_01"),
    ]
    t = _transcript(words, duration=3.0)
    plan = declutter.declutter(t, level="light")
    assert all(d.action == "keep" for d in plan.decisions)


def test_dead_air_protected_by_overlapping_audio_event():
    words = [Word(text="one", start=0.0, end=1.0), Word(text="two", start=2.0, end=3.0)]
    events = [AudioEvent(type="laughter", start=1.2, end=1.8)]
    t = _transcript(words, duration=3.0, audio_events=events)
    plan = declutter.declutter(t, level="light")
    assert all(d.action == "keep" for d in plan.decisions)


# ---------------------------------------------------------------------------
# _merge_spans (direct unit test of the merge helper)
# ---------------------------------------------------------------------------

def test_merge_spans_combines_overlapping_and_touching():
    spans = [(0.0, 1.0, "filler", "um"), (1.0, 2.0, "dead_air", ""), (5.0, 6.0, "filler", "uh")]
    merged = declutter._merge_spans(spans)
    assert merged == [(0.0, 2.0, "filler", "um"), (5.0, 6.0, "filler", "uh")]


def test_merge_spans_empty_input():
    assert declutter._merge_spans([]) == []


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _base_transcript() -> Transcript:
    words = [
        Word(text="so", start=0.0, end=0.3),
        Word(text="the", start=0.3, end=0.5),
        Word(text="thing", start=0.5, end=0.8),
        Word(text="is", start=0.8, end=1.0),
    ]
    return _transcript(words, duration=10.0)  # generous duration so % guard has room


def test_llm_removals_unioned_with_heuristic(monkeypatch):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"removals": [{"start": 5.0, "end": 6.0, "reason": "retake", "text": "a retake"}]}

    monkeypatch.setattr(declutter, "get_llm_client", lambda: FakeClient())
    t = _base_transcript()
    plan = declutter.declutter(t, level="standard")

    removed = [d for d in plan.decisions if d.action == "remove"]
    assert any(d.reason == "retake" and d.start == pytest.approx(5.0) for d in removed)
    assert plan.method == "llm"


def test_llm_invalid_entries_are_dropped_not_fatal(monkeypatch):
    class PartlyBadClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "removals": [
                    "not a dict",
                    {"start": "nope", "end": 1.0},
                    {"start": 6.0, "end": 5.0},  # end <= start
                    {"start": 2.0, "end": 3.0, "reason": "retake"},
                ]
            }

    monkeypatch.setattr(declutter, "get_llm_client", lambda: PartlyBadClient())
    t = _base_transcript()
    plan = declutter.declutter(t, level="standard")

    removed = [d for d in plan.decisions if d.action == "remove" and d.reason == "retake"]
    assert len(removed) == 1
    assert removed[0].start == pytest.approx(2.0)


def test_llm_unknown_reason_defaults_to_retake(monkeypatch):
    class WeirdReasonClient:
        def complete_json(self, system, user, schema_hint):
            return {"removals": [{"start": 2.0, "end": 3.0, "reason": "something-else"}]}

    monkeypatch.setattr(declutter, "get_llm_client", lambda: WeirdReasonClient())
    t = _base_transcript()
    plan = declutter.declutter(t, level="standard")
    removed = [d for d in plan.decisions if d.action == "remove"]
    assert removed[0].reason == "retake"


def test_llm_exception_falls_back_to_heuristic_only(monkeypatch):
    class FailingClient:
        def complete_json(self, system, user, schema_hint):
            raise RuntimeError("network exploded")

    monkeypatch.setattr(declutter, "get_llm_client", lambda: FailingClient())
    words = [Word(text="um", start=0.0, end=0.3), Word(text="hi", start=0.5, end=0.8)]
    t = _transcript(words, duration=0.8)
    plan = declutter.declutter(t, level="light")

    assert plan.method == "heuristic"
    removed = [d for d in plan.decisions if d.action == "remove"]
    assert len(removed) == 1 and removed[0].reason == "filler"


def test_llm_plan_rejected_when_removal_exceeds_40_percent(monkeypatch):
    class OverAggressiveClient:
        def complete_json(self, system, user, schema_hint):
            # Removes 9 of 10 seconds -- way over the 40% guard.
            return {"removals": [{"start": 0.0, "end": 9.0, "reason": "retake"}]}

    monkeypatch.setattr(declutter, "get_llm_client", lambda: OverAggressiveClient())
    t = _base_transcript()
    plan = declutter.declutter(t, level="standard")

    assert plan.method == "heuristic"
    assert all(d.reason != "retake" for d in plan.decisions if d.action == "remove")


def test_llm_not_called_at_level_off(monkeypatch):
    def boom():
        raise AssertionError("get_llm_client should not be consulted when level='off'")

    monkeypatch.setattr(declutter, "get_llm_client", boom)
    t = _base_transcript()
    plan = declutter.declutter(t, level="off")
    assert plan.method == "identity"
