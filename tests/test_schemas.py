from pipeline.schemas import (
    Clip,
    Learnings,
    MediaAsset,
    PerformanceRecord,
    PipelineRun,
    Post,
    PublishResult,
    Transcript,
    TranscriptSegment,
    Word,
)


def test_word_roundtrip():
    w = Word(text="hello", start=0.0, end=0.4, confidence=0.98)
    w2 = Word.model_validate_json(w.model_dump_json())
    assert w2 == w


def test_transcript_full_text_and_words():
    seg1 = TranscriptSegment(
        id=0,
        start=0.0,
        end=1.0,
        text="hello world",
        words=[Word(text="hello", start=0.0, end=0.4), Word(text="world", start=0.5, end=1.0)],
    )
    seg2 = TranscriptSegment(id=1, start=1.0, end=1.5, text="", words=[])
    t = Transcript(run_id="r1", source_path="a.mp4", duration=1.5, segments=[seg1, seg2])

    assert t.full_text() == "hello world"
    assert len(t.all_words()) == 2
    assert t.all_words()[1].text == "world"


def test_transcript_empty_segments():
    t = Transcript(run_id="r1", source_path="a.mp4", duration=0.0, segments=[])
    assert t.full_text() == ""
    assert t.all_words() == []


def test_clip_duration():
    c = Clip(id="c1", start=10.0, end=42.5, hook="h", topic="t", score=0.9)
    assert c.duration == 32.5
    assert c.selection_method == "heuristic"


def test_post_defaults():
    p = Post(platform="x", text="hello")
    assert p.hashtags == []
    assert p.generation_method == "template"


def test_media_asset_defaults():
    m = MediaAsset(run_id="r1", source="video.mp4", local_path="/tmp/video.mp4")
    assert m.duration == 0.0
    assert m.title is None


def test_pipeline_run_mark_done_is_idempotent():
    run = PipelineRun(run_id="r1", source="video.mp4", status="pending")
    run.mark_done("ingest")
    run.mark_done("ingest")
    assert run.stages_completed == ["ingest"]


def test_performance_record_engagement_rate_computed():
    rec = PerformanceRecord(post_id="p1", platform="x", views=100, likes=10, comments=5, shares=5)
    assert rec.compute_engagement_rate() == 0.2


def test_performance_record_engagement_rate_zero_views():
    rec = PerformanceRecord(post_id="p1", platform="x", views=0, likes=1)
    assert rec.compute_engagement_rate() == 0.0


def test_performance_record_engagement_rate_explicit_wins():
    rec = PerformanceRecord(post_id="p1", platform="x", views=100, likes=10, engagement_rate=0.5)
    assert rec.compute_engagement_rate() == 0.5


def test_learnings_defaults():
    learnings = Learnings()
    assert learnings.top_keywords == []
    assert learnings.ideal_clip_length_range is None


def test_publish_result():
    r = PublishResult(platform="x", method="outbox", location="/tmp/outbox/x")
    assert r.status == "ready"
