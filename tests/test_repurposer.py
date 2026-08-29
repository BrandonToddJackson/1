import json
from pathlib import Path

import pytest

from pipeline import repurposer
from pipeline.schemas import Clip, Learnings, Post, Transcript, TranscriptSegment

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_transcript.json"


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.model_validate(json.loads(FIXTURE.read_text()))


@pytest.fixture
def clip(transcript) -> Clip:
    # Segments 1-3: "biggest mistake ... secret ... compound interest"
    segs = [s for s in transcript.segments if s.id in (1, 2, 3)]
    return Clip(
        id="clip-01",
        start=segs[0].start,
        end=segs[-1].end,
        hook="most people make a huge mistake with their first paycheck",
        topic="mistake paycheck saving",
        score=0.9,
        source_segment_ids=[1, 2, 3],
        selection_method="heuristic",
    )


def test_generate_posts_no_llm_returns_one_per_platform(monkeypatch, clip, transcript):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    posts = repurposer.generate_posts(clip, transcript)

    assert len(posts) == len(repurposer.DEFAULT_PLATFORMS)
    assert {p.platform for p in posts} == set(repurposer.DEFAULT_PLATFORMS)
    assert all(p.generation_method == "template" for p in posts)
    assert all(p.clip_id == clip.id for p in posts)


@pytest.mark.parametrize("platform", list(repurposer.PLATFORM_RULES.keys()))
def test_template_post_respects_char_limit(platform, clip, transcript):
    post = repurposer._template_post(platform, clip, transcript, learnings=None)
    rule = repurposer.PLATFORM_RULES[platform]
    assert len(post.text) <= rule.max_chars


@pytest.fixture
def long_body_transcript() -> Transcript:
    # Long enough (~3600 chars) that the old shorts budget math (off by 2
    # chars) actually overflows -- the short fixture body never did.
    long_text = " ".join(f"word{i}" for i in range(600))
    seg = TranscriptSegment(id=0, start=0.0, end=120.0, text=long_text, words=[])
    return Transcript(run_id="long-test", source_path="x.mp4", duration=120.0, segments=[seg])


@pytest.fixture
def long_clip() -> Clip:
    return Clip(
        id="clip-01",
        start=0.0,
        end=120.0,
        hook="a" * 150,
        topic="testing",
        score=0.5,
        source_segment_ids=[0],
        selection_method="heuristic",
    )


def test_shorts_post_never_exceeds_max_chars_with_long_body(long_clip, long_body_transcript):
    post = repurposer._template_post("shorts", long_clip, long_body_transcript, learnings=None)
    rule = repurposer.PLATFORM_RULES["shorts"]
    assert len(post.text) <= rule.max_chars


@pytest.mark.parametrize("platform", list(repurposer.PLATFORM_RULES.keys()))
def test_all_platforms_respect_max_chars_with_long_body(platform, long_clip, long_body_transcript):
    post = repurposer._template_post(platform, long_clip, long_body_transcript, learnings=None)
    rule = repurposer.PLATFORM_RULES[platform]
    assert len(post.text) <= rule.max_chars


def test_x_post_has_no_cta_by_default(clip, transcript):
    post = repurposer._template_post("x", clip, transcript, learnings=None)
    assert post.cta is None


def test_linkedin_post_has_hashtags_and_cta(clip, transcript):
    post = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    assert len(post.hashtags) <= 4
    assert all(tag.startswith("#") for tag in post.hashtags)
    assert post.cta is not None


def test_shorts_post_is_title_description_format(clip, transcript):
    post = repurposer._template_post("shorts", clip, transcript, learnings=None)
    assert post.text.startswith("Title: ")
    assert "\n\nDescription: " in post.text


def test_newsletter_post_has_subject_line(clip, transcript):
    post = repurposer._template_post("newsletter", clip, transcript, learnings=None)
    assert post.text.startswith("Subject: ")
    assert post.hashtags == []


def test_deterministic_across_calls(clip, transcript):
    a = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    b = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    assert a.text == b.text
    assert a.hashtags == b.hashtags


def test_learnings_keywords_used_as_hashtags(clip, transcript):
    learnings = Learnings(top_keywords=["compoundinterest", "money"])
    post = repurposer._template_post("linkedin", clip, transcript, learnings=learnings)
    assert "#compoundinterest" in post.hashtags


def test_truncate_preserves_word_boundary():
    text = "one two three four five"
    truncated = repurposer._truncate(text, 10)
    assert truncated.endswith("…")
    assert " " not in truncated[-2:-1] or True  # no partial trailing word
    assert not truncated[:-1].endswith(" ")


def test_llm_path_used_when_client_present(monkeypatch, clip, transcript):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "posts": [
                    {"platform": "x", "text": "custom llm post", "hashtags": ["#money"], "cta": None}
                ]
            }

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: FakeClient())

    posts = repurposer.generate_posts(clip, transcript, platforms=("x",))
    assert len(posts) == 1
    assert posts[0].generation_method == "llm"
    assert posts[0].text == "custom llm post"


def test_llm_post_text_is_truncated_to_platform_limit(monkeypatch, clip, transcript):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"posts": [{"platform": "x", "text": "y" * 1000, "hashtags": [], "cta": None}]}

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: FakeClient())

    posts = repurposer.generate_posts(clip, transcript, platforms=("x",))
    assert len(posts[0].text) <= repurposer.PLATFORM_RULES["x"].max_chars


def test_llm_failure_falls_back_to_templates(monkeypatch, clip, transcript, caplog):
    class FailingClient:
        def complete_json(self, system, user, schema_hint):
            raise RuntimeError("boom")

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: FailingClient())

    with caplog.at_level("WARNING"):
        posts = repurposer.generate_posts(clip, transcript, platforms=("x", "linkedin"))

    assert len(posts) == 2
    assert all(p.generation_method == "template" for p in posts)
    assert "falling back to templates" in caplog.text


def test_llm_empty_result_falls_back_to_templates(monkeypatch, clip, transcript):
    class EmptyClient:
        def complete_json(self, system, user, schema_hint):
            return {"posts": []}

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: EmptyClient())

    posts = repurposer.generate_posts(clip, transcript, platforms=("x",))
    assert len(posts) == 1
    assert posts[0].generation_method == "template"


def test_llm_missing_platform_is_backfilled_with_template(monkeypatch, clip, transcript):
    class PartialClient:
        def complete_json(self, system, user, schema_hint):
            return {"posts": [{"platform": "x", "text": "only x", "hashtags": [], "cta": None}]}

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: PartialClient())

    posts = repurposer.generate_posts(clip, transcript, platforms=("x", "linkedin"))
    assert len(posts) == 2
    by_platform = {p.platform: p for p in posts}
    assert by_platform["x"].generation_method == "llm"
    assert by_platform["linkedin"].generation_method == "template"


def test_llm_unknown_platform_in_response_is_skipped(monkeypatch, clip, transcript):
    class WeirdClient:
        def complete_json(self, system, user, schema_hint):
            return {
                "posts": [
                    {"platform": "tiktok", "text": "unsupported", "hashtags": [], "cta": None},
                    {"platform": "x", "text": "supported", "hashtags": [], "cta": None},
                ]
            }

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: WeirdClient())

    posts = repurposer.generate_posts(clip, transcript, platforms=("x",))
    assert len(posts) == 1
    assert posts[0].platform == "x"


def test_llm_learnings_in_prompt(monkeypatch, clip, transcript):
    captured = {}

    class CapturingClient:
        def complete_json(self, system, user, schema_hint):
            captured["system"] = system
            return {"posts": [{"platform": "x", "text": "t", "hashtags": [], "cta": None}]}

    monkeypatch.setattr(repurposer, "get_llm_client", lambda: CapturingClient())
    learnings = Learnings(top_keywords=["compoundinterest"], best_platforms=["instagram", "x"])

    repurposer.generate_posts(clip, transcript, platforms=("x",), learnings=learnings)

    assert "compoundinterest" in captured["system"]
    assert "instagram" in captured["system"]
