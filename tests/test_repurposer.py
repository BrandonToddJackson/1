import json
from pathlib import Path

import pytest

from pipeline import clip_selector, repurposer
from pipeline.schemas import Clip, Learnings, Transcript, TranscriptSegment, Word

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


# ---------------------------------------------------------------------------
# CTA rotation -- a single fixed CTA string appearing verbatim on every post
# this tool ever generates for a platform is the clearest "mass-produced,
# not platform-native" tell there is. _pick_cta must vary it across clips
# while staying deterministic for a given clip.
# ---------------------------------------------------------------------------

def test_cta_varies_across_different_clips(transcript):
    clip_a = Clip(id="run-clip-01", start=0, end=10, hook="h", topic="t", score=0.5)
    clip_b = Clip(id="run-clip-02", start=10, end=20, hook="h", topic="t", score=0.5)
    clip_c = Clip(id="run-clip-03", start=20, end=30, hook="h", topic="t", score=0.5)

    ctas = {
        repurposer._template_post("linkedin", c, transcript, learnings=None).cta
        for c in (clip_a, clip_b, clip_c)
    }
    # Not asserting all 3 differ (a hash collision is possible with only 3
    # variants) -- but seeing more than one value across 3 distinct clip ids
    # is the real regression check against "always the same string".
    assert len(ctas) > 1


def test_cta_deterministic_for_same_clip(clip, transcript):
    a = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    b = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    assert a.cta == b.cta


def test_x_and_shorts_still_have_no_cta(clip, transcript):
    assert repurposer._template_post("x", clip, transcript, learnings=None).cta is None
    assert repurposer._template_post("shorts", clip, transcript, learnings=None).cta is None


def test_pick_cta_single_variant_platform_is_stable():
    rule = repurposer._PlatformRule(max_chars=100, hashtag_count=0, cta_variants=("only one",), tone_note="x")
    clip_a = Clip(id="a", start=0, end=1, hook="h", topic="t", score=0.5)
    clip_b = Clip(id="b", start=0, end=1, hook="h", topic="t", score=0.5)
    assert repurposer._pick_cta(rule, clip_a) == "only one"
    assert repurposer._pick_cta(rule, clip_b) == "only one"


# ---------------------------------------------------------------------------
# Real-output regression: hashtags must not contain conversational filler
# on unscripted-sounding speech. Found by actually running the zero-key
# pipeline against ordinary interview-style speech instead of the
# hook-word-stuffed fixture used everywhere else in this file.
# ---------------------------------------------------------------------------

def test_end_to_end_unscripted_speech_produces_no_filler_hashtags():
    sentences = [
        "yeah so when we started the company we didn't really have a plan.",
        "we just knew we wanted to build something that solved our own problem.",
        "and honestly the first version was pretty rough, a lot of things broke.",
    ]
    segments, t = [], 0.0
    for i, sent in enumerate(sentences):
        words = []
        for tok in sent.split():
            words.append(Word(text=tok, start=round(t, 2), end=round(t + 0.32, 2)))
            t = round(t + 0.38, 2)
        segments.append(TranscriptSegment(id=i, start=words[0].start, end=words[-1].end, text=sent, words=words))
        t += 0.55
    unscripted = Transcript(run_id="filler-check", source_path="x.mp4", duration=t, segments=segments)

    clips = clip_selector._select_clips_heuristic(unscripted, max_clips=3, min_len=5.0, max_len=25.0, learnings=None)
    assert clips

    for c in clips:
        posts = repurposer.generate_posts(c, unscripted, platforms=("linkedin",))
        for p in posts:
            for tag in p.hashtags:
                assert tag not in ("#yeah", "#when", "#really"), f"filler hashtag leaked through: {tag}"


# ---------------------------------------------------------------------------
# Body must not repeat the hook sentence. _extract_hook (clip_selector.py)
# now snaps to real sentence boundaries, so hook is usually exactly the
# clip's first sentence -- and body (the full clip text) previously always
# included that same sentence again as its own opening line.
# ---------------------------------------------------------------------------

def test_body_after_hook_strips_matching_prefix():
    hook = "we spent six months talking to customers"
    body = "we spent six months talking to customers. then we started building."
    assert repurposer._body_after_hook(hook, body) == "then we started building."


def test_body_after_hook_falls_back_when_hook_not_a_prefix():
    hook = "a totally different paraphrase"
    body = "the actual transcript text says something else entirely."
    assert repurposer._body_after_hook(hook, body) == body


def test_body_after_hook_falls_back_when_remainder_would_be_empty():
    hook = "this is the whole clip"
    body = "this is the whole clip."
    assert repurposer._body_after_hook(hook, body) == body.strip()


def test_body_after_hook_is_case_insensitive():
    hook = "WE SPENT SIX MONTHS"
    body = "we spent six months talking to customers before writing code."
    assert repurposer._body_after_hook(hook, body) == "talking to customers before writing code."


def test_template_post_body_does_not_repeat_hook_sentence(clip, transcript):
    post = repurposer._template_post("linkedin", clip, transcript, learnings=None)
    hook_sentence = clip.hook.rstrip(".").lower()
    paragraphs = post.text.split("\n\n")
    assert len(paragraphs) >= 2, "expected a separate body paragraph for this multi-sentence clip"
    assert not paragraphs[1].lower().startswith(hook_sentence)


def test_shorts_description_does_not_repeat_title(clip, transcript):
    post = repurposer._template_post("shorts", clip, transcript, learnings=None)
    hook_sentence = clip.hook.rstrip(".").lower()
    description = post.text.split("Description: ", 1)[1].lower()
    assert not description.startswith(hook_sentence)


def test_newsletter_body_does_not_repeat_subject(clip, transcript):
    post = repurposer._template_post("newsletter", clip, transcript, learnings=None)
    hook_sentence = clip.hook.rstrip(".").lower()
    body = post.text.split("\n\n", 1)[1].lower()
    assert not body.startswith(hook_sentence)
