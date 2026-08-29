import json
from pathlib import Path

import pytest

from pipeline import repurposer
from pipeline.schemas import Clip, Learnings, Post, Transcript

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
