"""End-to-end wiring test for the CLI, using a synthetic ffmpeg test video and
a stubbed transcribe stage (real faster-whisper is exercised separately in
the manual smoke test -- see README.md -- since it needs a model download)."""

import shutil
import subprocess

import pytest
from typer.testing import CliRunner

from pipeline import cli, transcribe as transcribe_module
from pipeline.schemas import Transcript, TranscriptSegment, Word
from pipeline.storage import load_run_state, run_dir

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

runner = CliRunner()


@pytest.fixture
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("cli-media") / "sample.mp4"
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


def test_full_run_end_to_end(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(transcribe_module, "transcribe", _fake_transcript)
    monkeypatch.setattr(cli, "transcribe_", transcribe_module)

    result = runner.invoke(cli.app, ["run", str(sample_video), "--min-len", "1", "--max-clips", "2"])

    assert result.exit_code == 0, result.output
    assert "done" in result.output

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_id = run_dirs[0].name

    run = load_run_state(run_id)
    assert run.status == "completed"
    assert set(run.stages_completed) == {
        "ingest", "enhance", "transcribe", "declutter", "select_clips", "cut", "caption", "graphics",
        "repurpose", "publish",
    }

    outbox = run_dir(run_id) / "outbox"
    assert outbox.exists()
    caption_files = list(outbox.rglob("caption.txt"))
    assert caption_files, "expected at least one outbox caption.txt"

    media_files = list(outbox.rglob("media.mp4"))
    assert media_files, "expected captioned clip media copied into the outbox"


def test_rerun_with_same_run_id_is_idempotent(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(transcribe_module, "transcribe", _fake_transcript)
    monkeypatch.setattr(cli, "transcribe_", transcribe_module)

    first = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", "fixed-run", "--min-len", "1"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(cli.app, ["run", str(sample_video), "--run-id", "fixed-run", "--min-len", "1"])
    assert second.exit_code == 0, second.output
    assert "resuming" in second.output
