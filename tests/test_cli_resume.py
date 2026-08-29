"""Regression tests for the cli.py resumability rewrite: stage invalidation,
--force/param-mismatch handling, incremental cut/caption retry-by-skipping,
fail-fast platform validation, and PipelineRun.status/.error actually being
set on a stage failure (none of this existed before -- see the plan's
Step 8)."""

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipeline import cli, cutter, repurposer
from pipeline import ingest as ingest_module
from pipeline import transcribe as transcribe_module
from pipeline.schemas import Clip, Post, Transcript, TranscriptSegment, Word
from pipeline.storage import load_run_state, read_json_list, run_dir, stage_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

runner = CliRunner()


@pytest.fixture
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli-resume-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def _fake_transcript(media_path, run_id, model_size=None) -> Transcript:
    sentences = [
        "the biggest mistake people make is waiting to invest.",
        "here is the secret nobody tells you about saving early.",
        "so start today because time is genuinely your best asset.",
    ]
    segments = []
    t = 0.0
    for i, sent in enumerate(sentences):
        words = []
        for tok in sent.split():
            words.append(Word(text=tok, start=round(t, 2), end=round(t + 0.3, 2)))
            t = round(t + 0.35, 2)
        segments.append(TranscriptSegment(id=i, start=words[0].start, end=words[-1].end, text=sent, words=words))
        t += 0.6
    return Transcript(run_id=run_id, source_path=str(media_path), duration=t, segments=segments, model="stub")


def _use_fake_transcribe(monkeypatch):
    monkeypatch.setattr(transcribe_module, "transcribe", _fake_transcript)
    monkeypatch.setattr(cli, "transcribe_", transcribe_module)


# --------------------------------------------------------------------------
# 1. Resume after a failed ingest (the flagship bug: README promised this
#    worked, it crashed with FileNotFoundError on media.json)
# --------------------------------------------------------------------------

def test_resume_after_failed_ingest(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "fail-ingest"
    real_ingest_fn = ingest_module.ingest

    def failing_ingest(source, rid):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(ingest_module, "ingest", failing_ingest)
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert first.exit_code == 1

    run = load_run_state(run_id)
    assert run.status == "failed"
    assert run.error.startswith("ingest:")
    assert not stage_path(run_id, "media").exists()

    monkeypatch.setattr(ingest_module, "ingest", real_ingest_fn)
    second = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert second.exit_code == 0, second.output

    run = load_run_state(run_id)
    assert run.status == "completed"
    assert set(run.stages_completed) == set(cli.STAGE_ORDER)


# --------------------------------------------------------------------------
# 2. Resume after a failed transcribe must NOT re-download/re-ingest
# --------------------------------------------------------------------------

def test_resume_after_failed_transcribe_does_not_reingest(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    ingest_calls = {"count": 0}
    real_ingest_fn = ingest_module.ingest

    def counting_ingest(source, rid):
        ingest_calls["count"] += 1
        return real_ingest_fn(source, rid)

    monkeypatch.setattr(ingest_module, "ingest", counting_ingest)

    transcribe_state = {"calls": 0}

    def flaky_transcribe(media_path, rid, model_size=None):
        transcribe_state["calls"] += 1
        if transcribe_state["calls"] == 1:
            raise RuntimeError("simulated transcribe failure")
        return _fake_transcript(media_path, rid, model_size)

    monkeypatch.setattr(transcribe_module, "transcribe", flaky_transcribe)
    monkeypatch.setattr(cli, "transcribe_", transcribe_module)

    run_id = "flaky-transcribe"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert first.exit_code == 1

    second = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert second.exit_code == 0, second.output
    assert ingest_calls["count"] == 1


# --------------------------------------------------------------------------
# 3. select-clips re-run invalidates every downstream artifact
# --------------------------------------------------------------------------

def test_select_clips_rerun_invalidates_downstream(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "reselect"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert first.exit_code == 0, first.output

    reselect = runner.invoke(cli.app, ["select-clips", run_id, "--min-len", "2", "--max-clips", "1"])
    assert reselect.exit_code == 0, reselect.output

    run = load_run_state(run_id)
    assert run.stages_completed == ["ingest", "enhance", "transcribe", "declutter", "select_clips"]

    base = run_dir(run_id)
    for stale in ("raw_clips.json", "captioned_clips.json", "posts.json", "publish_results.json", "clips_raw", "clips_captioned", "outbox"):
        assert not (base / stale).exists(), f"{stale} should have been invalidated"


# --------------------------------------------------------------------------
# 4. cut without clips.json fails clearly, no traceback / KeyError
# --------------------------------------------------------------------------

def test_cut_without_clips_reports_clear_error(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "no-clips-yet"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0
    assert runner.invoke(cli.app, ["enhance", run_id, "--off"]).exit_code == 0
    assert runner.invoke(cli.app, ["transcribe", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["cut", run_id])
    assert result.exit_code == 1
    assert "clips.json" in result.output
    assert "Traceback" not in result.output


# --------------------------------------------------------------------------
# 5. Re-running `run` with different tuning params requires --force
# --------------------------------------------------------------------------

def test_rerun_with_changed_max_clips_requires_force(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "param-change"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert first.exit_code == 0, first.output

    no_force = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "1"])
    assert no_force.exit_code == 1
    assert "--force" in no_force.output
    assert "max_clips" in no_force.output

    forced = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "1", "--force"])
    assert forced.exit_code == 0, forced.output

    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    assert len(clips) == 1


# --------------------------------------------------------------------------
# 6. --force only invalidates from the earliest AFFECTED stage -- changing
#    only --platforms must not discard cuts/captions
# --------------------------------------------------------------------------

def test_rerun_with_changed_platforms_only_invalidates_repurpose(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "platform-change"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert first.exit_code == 0, first.output

    clips_raw_dir = run_dir(run_id) / "clips_raw"
    captioned_path = stage_path(run_id, "captioned_clips")
    raw_mtime_before = clips_raw_dir.stat().st_mtime
    captioned_mtime_before = captioned_path.stat().st_mtime

    second = runner.invoke(
        cli.app,
        ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2", "--platforms", "x", "--force"],
    )
    assert second.exit_code == 0, second.output

    assert clips_raw_dir.exists() and clips_raw_dir.stat().st_mtime == raw_mtime_before
    assert captioned_path.exists() and captioned_path.stat().st_mtime == captioned_mtime_before

    posts = read_json_list(stage_path(run_id, "posts"), Post)
    assert posts
    assert all(p.platform == "x" for p in posts)


# --------------------------------------------------------------------------
# 7. A different source on the same run_id also requires --force
# --------------------------------------------------------------------------

def test_rerun_with_different_source_requires_force(monkeypatch, tmp_path, tmp_path_factory, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "source-change"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert first.exit_code == 0, first.output

    other_video = tmp_path_factory.mktemp("other-media") / "other.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=5",
            "-shortest", str(other_video),
        ],
        check=True, capture_output=True, text=True,
    )

    second = runner.invoke(cli.app, ["run", str(other_video), "--run-id", run_id, "--min-len", "1"])
    assert second.exit_code == 1
    assert "--force" in second.output
    assert "source" in second.output


# --------------------------------------------------------------------------
# 8. A completed run re-invoked with identical params is a clean no-op
# --------------------------------------------------------------------------

def test_completed_run_same_params_is_noop(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "already-done"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert first.exit_code == 0, first.output

    posts_path = stage_path(run_id, "posts")
    mtime_before = posts_path.stat().st_mtime

    second = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1"])
    assert second.exit_code == 0, second.output
    assert "nothing to do" in second.output
    assert posts_path.stat().st_mtime == mtime_before


# --------------------------------------------------------------------------
# 9. Unknown --platforms is rejected before any run directory is created
# --------------------------------------------------------------------------

def test_unknown_platform_rejected_before_ingest(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    result = runner.invoke(cli.app, ["run", str(sample_video), "--platforms", "linkedin,tiktok"])
    assert result.exit_code != 0
    assert "tiktok" in result.output
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# 10. A partial cut failure preserves already-cut clips; retry only redoes
#     the one that failed
# --------------------------------------------------------------------------

def test_partial_cut_failure_preserves_completed_clips(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "cut-partial"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0
    assert runner.invoke(cli.app, ["enhance", run_id, "--off"]).exit_code == 0
    assert runner.invoke(cli.app, ["transcribe", run_id]).exit_code == 0
    assert runner.invoke(cli.app, ["declutter", run_id]).exit_code == 0
    # A small --max-len forces the 3-sentence fake transcript to split into
    # multiple clips instead of one window covering everything.
    select_result = runner.invoke(cli.app, ["select-clips", run_id, "--min-len", "1", "--max-len", "4", "--max-clips", "2"])
    assert select_result.exit_code == 0, select_result.output

    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    assert len(clips) >= 2, "need at least 2 clips for this test to be meaningful"

    real_cut_clip = cutter.cut_clip
    call_log: list[str] = []

    def flaky_cut_clip(source, clip, out_dir):
        call_log.append(clip.id)
        if clip.id == clips[1].id:
            raise RuntimeError("simulated ffmpeg failure")
        return real_cut_clip(source, clip, out_dir)

    monkeypatch.setattr(cutter, "cut_clip", flaky_cut_clip)
    first_cut = runner.invoke(cli.app, ["cut", run_id])
    assert first_cut.exit_code == 1

    raw_map = cli._load_path_map(stage_path(run_id, "raw_clips"))
    assert clips[0].id in raw_map
    assert clips[1].id not in raw_map

    call_log.clear()

    def logging_cut_clip(source, clip, out_dir):
        call_log.append(clip.id)
        return real_cut_clip(source, clip, out_dir)

    monkeypatch.setattr(cutter, "cut_clip", logging_cut_clip)
    second_cut = runner.invoke(cli.app, ["cut", run_id])
    assert second_cut.exit_code == 0, second_cut.output
    assert call_log == [clips[1].id]  # only the previously-failed clip was retried


# --------------------------------------------------------------------------
# 11. Zero selected clips is a visible warning, not silent "success"
# --------------------------------------------------------------------------

def test_zero_clips_warns_and_summarizes(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    result = runner.invoke(cli.app, ["run", str(sample_video), "--min-len", "999"])
    assert result.exit_code == 0, result.output
    assert "no clips selected" in result.output
    assert "0 clips" in result.output


# --------------------------------------------------------------------------
# 12. A failing stage actually sets PipelineRun.status/.error
# --------------------------------------------------------------------------

def test_failed_stage_sets_status_and_error(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    def failing_generate_posts(*args, **kwargs):
        raise RuntimeError("simulated repurpose failure")

    monkeypatch.setattr(repurposer, "generate_posts", failing_generate_posts)

    result = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", "fail-repurpose", "--min-len", "1"])
    assert result.exit_code == 1

    run = load_run_state("fail-repurpose")
    assert run.status == "failed"
    assert run.error.startswith("repurpose:")


# --------------------------------------------------------------------------
# 13. Stage functions are plain and directly callable (no typer.OptionInfo
#     leaking in when called outside of a Typer command invocation)
# --------------------------------------------------------------------------

def test_stage_functions_are_callable_directly(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "direct-call"
    cli._stage_ingest(str(sample_video), run_id)
    cli._stage_enhance(run_id, enabled=False)
    cli._stage_transcribe(run_id)
    cli._stage_declutter(run_id, level="off")
    cli._stage_select_clips(run_id, max_clips=2, min_len=1.0, max_len=90.0)

    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    assert clips
    assert len(clips) <= 2


# --------------------------------------------------------------------------
# 14. enhance stage: identity artifact when disabled, transcribe reads from
#     enhanced_media.json (not media.json), --no-enhance wires through `run`
# --------------------------------------------------------------------------

def test_enhance_off_writes_identity_artifact_pointing_at_source(monkeypatch, tmp_path, sample_video):
    from pipeline.schemas import MediaAsset
    from pipeline.storage import read_json

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run_id = "enhance-off"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["enhance", run_id, "--off"])
    assert result.exit_code == 0, result.output

    original = read_json(stage_path(run_id, "media"), MediaAsset)
    enhanced = read_json(stage_path(run_id, "enhanced_media"), MediaAsset)
    assert enhanced.local_path == original.local_path
    assert enhanced.enhanced_from is None
    assert enhanced.loudness_lufs is None


def test_enhance_default_produces_denoised_media_with_loudness(monkeypatch, tmp_path, sample_video):
    from pipeline.schemas import MediaAsset
    from pipeline.storage import read_json

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run_id = "enhance-on"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["enhance", run_id])
    assert result.exit_code == 0, result.output

    original = read_json(stage_path(run_id, "media"), MediaAsset)
    enhanced = read_json(stage_path(run_id, "enhanced_media"), MediaAsset)
    assert enhanced.local_path != original.local_path
    assert enhanced.enhanced_from == original.local_path
    assert Path(enhanced.local_path).exists()


def test_transcribe_requires_enhanced_media_not_media(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "needs-enhance-first"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["transcribe", run_id])
    assert result.exit_code == 1
    assert "enhanced_media.json" in result.output
    assert "enhance" in result.output


def test_run_no_enhance_flag_skips_enhancement_end_to_end(monkeypatch, tmp_path, sample_video):
    from pipeline.schemas import MediaAsset
    from pipeline.storage import read_json

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "no-enhance-run"
    result = runner.invoke(
        cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--no-enhance"],
    )
    assert result.exit_code == 0, result.output

    run = load_run_state(run_id)
    assert "enhance" in run.stages_completed
    assert run.params["enhance"] is False

    original = read_json(stage_path(run_id, "media"), MediaAsset)
    enhanced = read_json(stage_path(run_id, "enhanced_media"), MediaAsset)
    assert enhanced.local_path == original.local_path


def test_select_clips_rerun_does_not_invalidate_enhance(monkeypatch, tmp_path, sample_video):
    """select-clips invalidates from itself onward -- enhance/transcribe,
    which run BEFORE it, must survive untouched."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "reselect-keeps-enhance"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert first.exit_code == 0, first.output

    enhanced_mtime_before = stage_path(run_id, "enhanced_media").stat().st_mtime

    reselect = runner.invoke(cli.app, ["select-clips", run_id, "--min-len", "2", "--max-clips", "1"])
    assert reselect.exit_code == 0, reselect.output

    assert stage_path(run_id, "enhanced_media").exists()
    assert stage_path(run_id, "enhanced_media").stat().st_mtime == enhanced_mtime_before


# --------------------------------------------------------------------------
# 15. declutter stage: default off produces an identity clean transcript,
#     select-clips requires transcript_clean.json (not transcript.json),
#     and re-running select-clips doesn't invalidate declutter.
# --------------------------------------------------------------------------

def test_declutter_default_off_produces_identity_clean_transcript(monkeypatch, tmp_path, sample_video):
    from pipeline.schemas import Transcript as TranscriptSchema
    from pipeline.storage import read_json

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "declutter-off"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0
    assert runner.invoke(cli.app, ["enhance", run_id, "--off"]).exit_code == 0
    assert runner.invoke(cli.app, ["transcribe", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["declutter", run_id])
    assert result.exit_code == 0, result.output

    original = read_json(stage_path(run_id, "transcript"), TranscriptSchema)
    clean = read_json(stage_path(run_id, "transcript_clean"), TranscriptSchema)
    assert [w.text for w in clean.all_words()] == [w.text for w in original.all_words()]
    assert clean.duration == original.duration

    from pipeline.schemas import EditPlan

    plan = read_json(stage_path(run_id, "edit_plan"), EditPlan)
    assert plan.method == "identity"
    assert plan.level == "off"


def test_select_clips_requires_transcript_clean_not_transcript(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "needs-declutter-first"
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", run_id]).exit_code == 0
    assert runner.invoke(cli.app, ["enhance", run_id, "--off"]).exit_code == 0
    assert runner.invoke(cli.app, ["transcribe", run_id]).exit_code == 0

    result = runner.invoke(cli.app, ["select-clips", run_id, "--min-len", "1"])
    assert result.exit_code == 1
    assert "transcript_clean.json" in result.output
    assert "declutter" in result.output


def test_select_clips_rerun_does_not_invalidate_declutter(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "reselect-keeps-declutter"
    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert first.exit_code == 0, first.output

    clean_mtime_before = stage_path(run_id, "transcript_clean").stat().st_mtime

    reselect = runner.invoke(cli.app, ["select-clips", run_id, "--min-len", "2", "--max-clips", "1"])
    assert reselect.exit_code == 0, reselect.output

    assert stage_path(run_id, "transcript_clean").exists()
    assert stage_path(run_id, "transcript_clean").stat().st_mtime == clean_mtime_before


def test_run_with_declutter_level_light_writes_edit_plan(monkeypatch, tmp_path, sample_video):
    from pipeline.schemas import EditPlan
    from pipeline.storage import read_json

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _use_fake_transcribe(monkeypatch)

    run_id = "declutter-light-run"
    result = runner.invoke(
        cli.app,
        ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--declutter-level", "light"],
    )
    assert result.exit_code == 0, result.output

    run = load_run_state(run_id)
    assert run.params["declutter_level"] == "light"

    plan = read_json(stage_path(run_id, "edit_plan"), EditPlan)
    assert plan.level == "light"
