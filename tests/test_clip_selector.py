import json
from pathlib import Path

import pytest

from pipeline import clip_selector
from pipeline.schemas import Clip, Learnings, Transcript, TranscriptSegment, Word

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_transcript.json"


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.model_validate(json.loads(FIXTURE.read_text()))


def test_uses_heuristic_path_with_no_llm_client(monkeypatch, transcript):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    clips = clip_selector.select_clips(transcript, max_clips=3, min_len=15.0, max_len=40.0)

    assert clips
    assert all(c.selection_method == "heuristic" for c in clips)


def test_respects_length_bounds(transcript):
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=5, min_len=15.0, max_len=35.0, learnings=None
    )
    assert clips
    for c in clips:
        assert 15.0 <= c.duration <= 35.0 + 0.01


def test_respects_max_clips(transcript):
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=2, min_len=10.0, max_len=90.0, learnings=None
    )
    assert len(clips) <= 2


def test_clips_do_not_overlap(transcript):
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=5, min_len=10.0, max_len=30.0, learnings=None
    )
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start + 1e-6


def test_clips_are_chronologically_ordered(transcript):
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=5, min_len=10.0, max_len=30.0, learnings=None
    )
    starts = [c.start for c in clips]
    assert starts == sorted(starts)


def test_scores_are_descending_before_reordering(transcript):
    # The highest-scored non-overlapping window should include the densest
    # cluster of hook words (the "biggest mistake ... secret ... compound
    # interest" section of the fixture).
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=1, min_len=15.0, max_len=40.0, learnings=None
    )
    assert len(clips) == 1
    assert any(kw in clips[0].hook.lower() or kw in clips[0].topic for kw in ("mistake", "secret", "biggest"))


def test_empty_transcript_returns_no_clips():
    empty = Transcript(run_id="r", source_path="x.mp4", duration=0.0, segments=[])
    assert clip_selector._select_clips_heuristic(empty, 5, 10.0, 60.0, None) == []


def test_learnings_keywords_influence_scoring(transcript):
    # Build the single candidate window covering the grocery-store sentence
    # and confirm injecting it as a learned keyword raises its score.
    words = transcript.all_words()
    grocery_words = [w for w in words if "grocery" in w.text.lower()]
    assert grocery_words
    boundaries = clip_selector._find_boundaries(words)
    candidates = clip_selector._build_candidates(transcript, boundaries, min_len=5.0, max_len=90.0)
    grocery_candidate = next(
        c
        for c in candidates
        if "grocery" in clip_selector._candidate_text(c, words).lower()
        and "vegetables" in clip_selector._candidate_text(c, words).lower()
    )

    baseline_score = clip_selector._score_candidate(grocery_candidate, words, clip_selector.DEFAULT_HOOK_WORDS, 30.0)
    boosted_hook_words = set(clip_selector.DEFAULT_HOOK_WORDS) | {"vegetables", "grocery"}
    boosted_score = clip_selector._score_candidate(grocery_candidate, words, boosted_hook_words, 30.0)

    assert boosted_score > baseline_score

    # And end-to-end: select_clips must actually merge learnings.top_keywords
    # into the hook-word set it scores with (not silently ignore them).
    learnings = Learnings(top_keywords=["vegetables", "grocery"])
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=len(candidates), min_len=5.0, max_len=90.0, learnings=learnings
    )
    grocery_seg_id = next(seg.id for seg in transcript.segments if "grocery" in seg.text.lower())

    def _find_clip_covering(clip_list):
        return next(c for c in clip_list if grocery_seg_id in c.source_segment_ids)

    grocery_clip = _find_clip_covering(clips)
    plain_clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=len(candidates), min_len=5.0, max_len=90.0, learnings=None
    )
    plain_grocery_clip = _find_clip_covering(plain_clips)
    assert grocery_clip.score > plain_grocery_clip.score


def test_llm_path_used_when_client_present(monkeypatch, transcript):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "clips": [
                    {"start": 0.0, "end": 20.0, "hook": "welcome back", "topic": "intro", "score": 0.8}
                ]
            }

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: FakeClient())

    clips = clip_selector.select_clips(transcript, max_clips=3)
    assert len(clips) == 1
    assert clips[0].selection_method == "llm"
    assert clips[0].hook == "welcome back"


def test_clip_ids_are_run_scoped_heuristic(transcript):
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=3, min_len=15.0, max_len=40.0, learnings=None
    )
    assert clips
    for c in clips:
        assert c.id.startswith(f"{transcript.run_id}-clip-")


def test_clip_ids_are_run_scoped_llm(monkeypatch, transcript):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"clips": [{"start": 0.0, "end": 20.0, "hook": "h", "topic": "t", "score": 0.8}]}

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: FakeClient())
    clips = clip_selector.select_clips(transcript, max_clips=3)
    assert clips[0].id == f"{transcript.run_id}-clip-01"


def test_llm_failure_falls_back_to_heuristic(monkeypatch, transcript, caplog):
    class FailingClient:
        def complete_json(self, system, user, schema_hint):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: FailingClient())

    with caplog.at_level("WARNING"):
        clips = clip_selector.select_clips(transcript, max_clips=3, min_len=15.0, max_len=40.0)

    assert clips
    assert all(c.selection_method == "heuristic" for c in clips)
    assert "falling back to heuristic" in caplog.text


def test_llm_empty_result_falls_back_to_heuristic(monkeypatch, transcript):
    class EmptyClient:
        def complete_json(self, system, user, schema_hint):
            return {"clips": []}

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: EmptyClient())

    clips = clip_selector.select_clips(transcript, max_clips=3, min_len=15.0, max_len=40.0)
    assert clips
    assert all(c.selection_method == "heuristic" for c in clips)


def test_llm_clips_clamped_to_transcript_duration(monkeypatch, transcript):
    class HallucinatingClient:
        def complete_json(self, system, user, schema_hint):
            return {"clips": [{"start": 0.0, "end": 99999.0, "hook": "h", "topic": "t", "score": 0.9}]}

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: HallucinatingClient())

    clips = clip_selector.select_clips(transcript, max_clips=3, min_len=5.0, max_len=40.0)
    assert clips
    for c in clips:
        assert c.end <= transcript.duration
        assert c.duration <= 40.0 + 0.01


def test_llm_malformed_entry_skipped_not_fatal(monkeypatch, transcript):
    class PartlyBadClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "clips": [
                    {"hook": "missing start/end entirely"},
                    {"start": 0.0, "end": 20.0, "hook": "good one", "topic": "t", "score": 0.9},
                ]
            }

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: PartlyBadClient())

    clips = clip_selector.select_clips(transcript, max_clips=3)
    assert len(clips) == 1
    assert clips[0].hook == "good one"


def test_llm_overlapping_clips_are_deduped(monkeypatch, transcript):
    class OverlappingClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "clips": [
                    {"start": 0.0, "end": 20.0, "hook": "a", "topic": "t", "score": 0.5},
                    {"start": 5.0, "end": 25.0, "hook": "b", "topic": "t", "score": 0.9},
                ]
            }

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: OverlappingClient())

    clips = clip_selector.select_clips(transcript, max_clips=5)
    assert len(clips) == 1
    assert clips[0].hook == "b"  # the higher-scored one wins


def test_llm_learnings_appear_in_prompt(monkeypatch, transcript):
    captured = {}

    class CapturingClient:
        def complete_json(self, system, user, schema_hint):
            captured["system"] = system
            return {"clips": [{"start": 0.0, "end": 20.0, "hook": "h", "topic": "t", "score": 0.8}]}

    monkeypatch.setattr(clip_selector, "get_llm_client", lambda: CapturingClient())
    learnings = Learnings(top_keywords=["compoundinterest"], ideal_clip_length_range=(30.0, 45.0))

    clip_selector.select_clips(transcript, max_clips=3, learnings=learnings)

    assert "compoundinterest" in captured["system"]
    assert "30-45s" in captured["system"]


def test_zero_ideal_clip_length_range_does_not_divide_by_zero(transcript):
    learnings = Learnings(ideal_clip_length_range=(0.0, 0.0))
    clips = clip_selector._select_clips_heuristic(
        transcript, max_clips=3, min_len=15.0, max_len=40.0, learnings=learnings
    )
    # Must not raise ZeroDivisionError; a real result isn't required.
    assert isinstance(clips, list)


def test_build_candidates_is_not_quadratic():
    """Regression test for the confirmed quadratic slowdown in
    _build_candidates (measured 0.11s@5min -> 1.75s@20min before the bisect
    fix). Generous wall-clock slack since this runs on shared CI hardware."""
    import time

    def _make_transcript(minutes: int) -> Transcript:
        segments = []
        t = 0.0
        seg_id = 0
        for _ in range(minutes * 20):  # ~20 short sentences per minute
            words = []
            for w in range(6):
                start = round(t, 2)
                end = round(t + 0.3, 2)
                text = f"word{w}" + ("." if w == 5 else "")  # only the last word ends the sentence
                words.append(Word(text=text, start=start, end=end))
                t = round(end + 0.05, 2)
            segments.append(
                TranscriptSegment(id=seg_id, start=words[0].start, end=words[-1].end, text="placeholder", words=words)
            )
            seg_id += 1
            t = round(t + 0.6, 2)  # inter-sentence pause -> boundary
        return Transcript(run_id="perf-test", source_path="x.mp4", duration=t, segments=segments)

    transcript_20min = _make_transcript(20)
    words = transcript_20min.all_words()
    boundaries = clip_selector._find_boundaries(words)

    start = time.monotonic()
    candidates = clip_selector._build_candidates(transcript_20min, boundaries, min_len=20.0, max_len=90.0)
    elapsed = time.monotonic() - start

    assert candidates
    assert elapsed < 1.5, f"_build_candidates took {elapsed:.2f}s on a 20-minute transcript (expected <1.5s)"
