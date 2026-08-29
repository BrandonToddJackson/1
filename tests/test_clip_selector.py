import json
from pathlib import Path

import pytest

from pipeline import clip_selector
from pipeline.schemas import Clip, Learnings, Transcript

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
    grocery_words = [w for w in transcript.all_words() if "grocery" in w.text.lower()]
    assert grocery_words
    boundaries = clip_selector._find_boundaries(transcript.all_words())
    candidates = clip_selector._build_candidates(transcript, boundaries, min_len=5.0, max_len=90.0)
    grocery_candidate = next(c for c in candidates if "grocery" in c.text.lower() and "vegetables" in c.text.lower())

    baseline_score = clip_selector._score_candidate(grocery_candidate, clip_selector.DEFAULT_HOOK_WORDS, 30.0)
    boosted_hook_words = set(clip_selector.DEFAULT_HOOK_WORDS) | {"vegetables", "grocery"}
    boosted_score = clip_selector._score_candidate(grocery_candidate, boosted_hook_words, 30.0)

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
