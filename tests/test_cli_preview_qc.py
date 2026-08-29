"""CLI wiring tests for the `preview` and `qc` non-stage commands (siblings
of `analyze`, not part of STAGE_ORDER) -- real ffmpeg against a synthetic
clip produced by a real `run`."""

import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from pipeline import cli, transcribe as transcribe_module
from pipeline.schemas import Transcript, TranscriptSegment, Word
from pipeline.storage import read_json, run_dir, stage_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

runner = CliRunner()


@pytest.fixture
def sample_video(tmp_path_factory):
    # 14s, comfortably longer than _fake_transcript's ~11.3s of generated
    # speech -- a video shorter than the transcript would make `cut`
    # truncate at the source's own length, which is a REAL condition
    # test_qc.py should (and does) flag, not something to avoid here.
    out = tmp_path_factory.mktemp("cli-preview-qc-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=14:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=14",
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


def _run_pipeline(monkeypatch, tmp_path, sample_video, run_id):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(transcribe_module, "transcribe", _fake_transcript)
    monkeypatch.setattr(cli, "transcribe_", transcribe_module)
    result = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", run_id, "--min-len", "1", "--max-clips", "2"])
    assert result.exit_code == 0, result.output


def test_preview_extracts_frames_for_all_clips(monkeypatch, tmp_path, sample_video):
    run_id = "preview-all"
    _run_pipeline(monkeypatch, tmp_path, sample_video, run_id)

    result = runner.invoke(cli.app, ["preview", run_id])
    assert result.exit_code == 0, result.output

    preview_dir = run_dir(run_id) / "preview"
    assert preview_dir.exists()
    clip_dirs = list(preview_dir.iterdir())
    assert clip_dirs
    for d in clip_dirs:
        pngs = list(d.glob("*.png"))
        assert pngs, f"expected at least one frame under {d}"


def test_preview_with_explicit_at_and_clip_id(monkeypatch, tmp_path, sample_video):
    run_id = "preview-explicit"
    _run_pipeline(monkeypatch, tmp_path, sample_video, run_id)

    from pipeline.storage import read_json_list
    from pipeline.schemas import Clip

    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    target = clips[0].id

    result = runner.invoke(cli.app, ["preview", run_id, "--clip-id", target, "--at", "0,0.5"])
    assert result.exit_code == 0, result.output

    frames = list((run_dir(run_id) / "preview" / target).glob("*.png"))
    assert len(frames) == 2


def test_preview_unknown_clip_id_fails_clearly(monkeypatch, tmp_path, sample_video):
    run_id = "preview-unknown"
    _run_pipeline(monkeypatch, tmp_path, sample_video, run_id)

    result = runner.invoke(cli.app, ["preview", run_id, "--clip-id", "nonexistent-clip"])
    assert result.exit_code == 1
    assert "unknown clip id" in result.output


def test_preview_requires_final_clips(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", "preview-early"]).exit_code == 0

    result = runner.invoke(cli.app, ["preview", "preview-early"])
    assert result.exit_code == 1
    assert "final_clips.json" in result.output


def test_qc_writes_report_and_passes_on_healthy_run(monkeypatch, tmp_path, sample_video):
    run_id = "qc-healthy"
    _run_pipeline(monkeypatch, tmp_path, sample_video, run_id)

    result = runner.invoke(cli.app, ["qc", run_id])
    assert result.exit_code == 0, result.output

    from pipeline.schemas import QCReport

    report = read_json(stage_path(run_id, "qc"), QCReport)
    assert report.run_id == run_id
    # Duration/silence/caption-cue/safe-area checks should all be clean --
    # video comfortably outlasts the transcript, no real gaps, short cues,
    # no graphics beats (no LLM key reachable in CI). Loudness is excluded
    # deliberately: a pure sine tone has ~zero dynamic range, a known
    # loudnorm edge case documented elsewhere in this repo (audio.py) where
    # ffmpeg can't reliably hit a target on synthetic single-frequency
    # audio -- real speech doesn't have this problem.
    non_loudness_warnings = [f for f in report.findings if f.severity == "warning" and f.check != "loudness"]
    assert non_loudness_warnings == []


def test_qc_requires_final_clips(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert runner.invoke(cli.app, ["ingest", str(sample_video), "--run-id", "qc-early"]).exit_code == 0

    result = runner.invoke(cli.app, ["qc", "qc-early"])
    assert result.exit_code == 1
    assert "final_clips.json" in result.output
