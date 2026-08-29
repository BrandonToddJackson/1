"""Tests for ingest/cutter/captioner that exercise the real ffmpeg binary
against a tiny synthetically generated video (no network dependency -- see
README.md's smoke-test section for the full ingest->transcribe->cut->caption
run against faster-whisper)."""

import shutil
import subprocess

import pytest

from pipeline import captioner, cutter, ingest
from pipeline.schemas import Clip, Word

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def test_is_url():
    assert ingest.is_url("https://youtube.com/watch?v=abc")
    assert ingest.is_url("http://example.com/video.mp4")
    assert not ingest.is_url("/home/user/video.mp4")
    assert not ingest.is_url("video.mp4")


def test_ingest_local_file(monkeypatch, tmp_path, sample_video):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    asset = ingest.ingest(str(sample_video), "run-1")

    assert asset.source == str(sample_video)
    assert (tmp_path / "run-1" / "source.mp4").exists()
    assert asset.duration == pytest.approx(6.0, abs=0.5)


def test_ingest_missing_local_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ingest.ingest("/nonexistent/path/video.mp4", "run-1")


def test_cut_clip_produces_file_with_expected_duration(tmp_path, sample_video):
    clip = Clip(id="clip-01", start=1.0, end=4.0, hook="h", topic="t", score=0.5)
    out_dir = tmp_path / "clips"

    out_path = cutter.cut_clip(sample_video, clip, out_dir)

    assert out_path.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    assert duration == pytest.approx(3.0, abs=0.4)


def test_cut_all_returns_one_path_per_clip(tmp_path, sample_video):
    clips = [
        Clip(id="clip-01", start=0.0, end=2.0, hook="h", topic="t", score=0.5),
        Clip(id="clip-02", start=2.0, end=4.0, hook="h", topic="t", score=0.5),
    ]
    paths = cutter.cut_all(sample_video, clips, tmp_path / "clips")
    assert set(paths.keys()) == {"clip-01", "clip-02"}
    assert all(p.exists() for p in paths.values())


def test_words_to_srt_groups_into_cues():
    words = [Word(text=f"word{i}", start=i * 0.3, end=i * 0.3 + 0.25) for i in range(8)]
    srt = captioner.words_to_srt(words, max_words_per_cue=4)
    assert srt.count("-->") == 2
    assert "word0" in srt and "word7" in srt


def test_words_to_srt_breaks_on_sentence_end():
    words = [
        Word(text="hello", start=0.0, end=0.3),
        Word(text="world.", start=0.3, end=0.6),
        Word(text="next", start=0.7, end=1.0),
    ]
    srt = captioner.words_to_srt(words, max_words_per_cue=10)
    assert srt.count("-->") == 2  # breaks after "world." even under max_words_per_cue


def test_words_to_srt_empty_returns_empty_string():
    assert captioner.words_to_srt([]) == ""


def test_words_to_srt_offset_shifts_timestamps():
    words = [Word(text="hi", start=10.0, end=10.5)]
    srt = captioner.words_to_srt(words, offset=10.0)
    assert "00:00:00,000 --> 00:00:00,500" in srt


def test_render_captioned_clip_burns_in_subtitles(tmp_path, sample_video):
    clip = Clip(id="clip-01", start=1.0, end=4.0, hook="h", topic="t", score=0.5)
    raw_clip = cutter.cut_clip(sample_video, clip, tmp_path / "raw")
    words = [Word(text="hello", start=1.2, end=1.6), Word(text="world.", start=1.7, end=2.1)]

    out_path = tmp_path / "captioned" / "clip-01.mp4"
    result = captioner.render_captioned_clip(raw_clip, words, out_path, offset=clip.start)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_captioned_clip_passes_through_when_no_words(tmp_path, sample_video):
    clip = Clip(id="clip-01", start=1.0, end=3.0, hook="h", topic="t", score=0.5)
    raw_clip = cutter.cut_clip(sample_video, clip, tmp_path / "raw")

    out_path = tmp_path / "captioned" / "clip-01.mp4"
    result = captioner.render_captioned_clip(raw_clip, [], out_path, offset=clip.start)

    assert result == out_path
    assert out_path.exists()
