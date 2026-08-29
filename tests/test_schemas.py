from pipeline.schemas import (
    AudioEvent,
    Clip,
    EditDecision,
    EditPlan,
    GraphicsBeat,
    GraphicsPlan,
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


def test_pipeline_run_params_default_empty():
    run = PipelineRun(run_id="r1", source="video.mp4")
    assert run.params == {}


def test_pipeline_run_loads_json_without_params_field():
    # Simulates a run.json written before `params` existed.
    old_json = (
        '{"run_id": "r1", "source": "video.mp4", "created_at": "2026-01-01T00:00:00Z", '
        '"stages_completed": ["ingest"], "status": "in_progress", "error": null}'
    )
    run = PipelineRun.model_validate_json(old_json)
    assert run.params == {}
    assert run.stages_completed == ["ingest"]


def test_pipeline_run_undo_removes_stage():
    run = PipelineRun(run_id="r1", source="video.mp4")
    run.mark_done("ingest")
    run.mark_done("transcribe")
    run.undo("transcribe")
    assert run.stages_completed == ["ingest"]


def test_pipeline_run_undo_missing_stage_is_noop():
    run = PipelineRun(run_id="r1", source="video.mp4")
    run.undo("transcribe")
    assert run.stages_completed == []


def test_pipeline_run_mark_failed():
    run = PipelineRun(run_id="r1", source="video.mp4", status="in_progress")
    run.mark_failed("cut", RuntimeError("ffmpeg exploded"))
    assert run.status == "failed"
    assert run.error == "cut: ffmpeg exploded"


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


def test_publish_result_blotato_method():
    r = PublishResult(platform="x", method="blotato", location="post-123", status="published")
    assert r.method == "blotato"


def test_word_speaker_defaults_none():
    w = Word(text="hi", start=0.0, end=0.5)
    assert w.speaker is None


def test_audio_event_roundtrip():
    e = AudioEvent(type="laughter", start=1.0, end=2.5)
    assert AudioEvent.model_validate_json(e.model_dump_json()) == e


def test_transcript_diarization_defaults():
    t = Transcript(run_id="r1", source_path="x.mp4", duration=10.0)
    assert t.speakers == ["SPEAKER_00"]
    assert t.diarization == "none"
    assert t.audio_events == []


def test_transcript_loads_json_without_diarization_fields():
    # Simulates a transcript.json written before diarization support existed.
    old_json = (
        '{"run_id": "r1", "source_path": "x.mp4", "language": "en", '
        '"duration": 10.0, "segments": [], "model": "faster-whisper-base"}'
    )
    t = Transcript.model_validate_json(old_json)
    assert t.speakers == ["SPEAKER_00"]
    assert t.diarization == "none"
    assert t.audio_events == []


def test_media_asset_enhance_fields_default_none():
    m = MediaAsset(run_id="r1", source="video.mp4", local_path="/tmp/video.mp4")
    assert m.enhanced_from is None
    assert m.loudness_lufs is None


def test_media_asset_loads_json_without_enhance_fields():
    old_json = '{"run_id": "r1", "source": "x.mp4", "local_path": "/tmp/x.mp4"}'
    m = MediaAsset.model_validate_json(old_json)
    assert m.enhanced_from is None


def test_edit_decision_defaults():
    d = EditDecision(start=0.0, end=1.0, action="keep")
    assert d.reason is None
    assert d.confidence == 1.0
    assert d.text == ""


def test_edit_plan_identity_default_method():
    plan = EditPlan(
        run_id="r1", source_duration=5.0,
        decisions=[EditDecision(start=0.0, end=5.0, action="keep")],
    )
    assert plan.method == "identity"
    assert plan.level == "off"


def test_graphics_beat_defaults():
    beat = GraphicsBeat(composition="lower-third")
    assert beat.variables == {}
    assert beat.duration == 0.0


def test_graphics_plan_default_skipped():
    plan = GraphicsPlan(clip_id="clip-01")
    assert plan.method == "skipped"
    assert plan.beats == []
    assert plan.skipped_reason is None


def test_graphics_plan_llm_method():
    plan = GraphicsPlan(
        clip_id="clip-01",
        method="llm",
        beats=[GraphicsBeat(composition="stat-card", variables={"value": "10x"}, anchor_word="ten")],
    )
    assert len(plan.beats) == 1
    assert plan.beats[0].composition == "stat-card"
