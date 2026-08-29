"""Real-ffmpeg tests for cutter.cut_ranges' multi-segment EDL cutting --
synthetic lavfi media, no network, no checked-in fixtures (matches
tests/test_media.py's pattern)."""

import shutil
import subprocess

import pytest

from pipeline import cutter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("cutter-edl") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory):
    """No audio stream at all -- exercises the video-only filtergraph path."""
    out = tmp_path_factory.mktemp("cutter-edl-silent") / "silent.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=15",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def _probe_duration(path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def _has_audio(path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def test_single_range_matches_old_single_cut_behavior(tmp_path, sample_video):
    out_path = tmp_path / "single.mp4"
    cutter.cut_ranges(sample_video, [(1.0, 4.0)], out_path)
    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(3.0, abs=0.4)


def test_multi_range_duration_is_sum_of_ranges(tmp_path, sample_video):
    out_path = tmp_path / "multi.mp4"
    # Two 2s ranges -- total kept content is 4s, regardless of the 5s gap
    # between them in the source.
    cutter.cut_ranges(sample_video, [(0.0, 2.0), (4.0, 6.0)], out_path)
    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(4.0, abs=0.4)


def test_three_ranges_duration_is_sum_of_ranges(tmp_path, sample_video):
    out_path = tmp_path / "three.mp4"
    cutter.cut_ranges(sample_video, [(0.0, 1.0), (3.0, 4.5), (7.0, 9.0)], out_path)
    assert out_path.exists()
    # 1.0 + 1.5 + 2.0 = 4.5s
    assert _probe_duration(out_path) == pytest.approx(4.5, abs=0.4)


def test_no_audio_source_produces_video_only_output(tmp_path, silent_video):
    out_path = tmp_path / "silent-cut.mp4"
    cutter.cut_ranges(silent_video, [(0.0, 2.0), (3.0, 5.0)], out_path)
    assert out_path.exists()
    assert not _has_audio(out_path)
    assert _probe_duration(out_path) == pytest.approx(4.0, abs=0.4)


def test_output_has_audio_when_source_has_audio(tmp_path, sample_video):
    out_path = tmp_path / "with-audio.mp4"
    cutter.cut_ranges(sample_video, [(0.0, 2.0), (4.0, 6.0)], out_path)
    assert _has_audio(out_path)


def test_empty_ranges_raises_value_error(tmp_path, sample_video):
    with pytest.raises(ValueError):
        cutter.cut_ranges(sample_video, [], tmp_path / "out.mp4")


def test_zero_length_ranges_are_filtered_out(tmp_path, sample_video):
    out_path = tmp_path / "with-zero.mp4"
    # A zero-length range mixed in with real ones must not break the graph.
    cutter.cut_ranges(sample_video, [(1.0, 1.0), (0.0, 2.0)], out_path)
    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(2.0, abs=0.4)


def test_all_zero_length_ranges_raises_value_error(tmp_path, sample_video):
    with pytest.raises(ValueError):
        cutter.cut_ranges(sample_video, [(1.0, 1.0), (2.0, 2.0)], tmp_path / "out.mp4")


def test_many_ranges_get_capped_by_max_keep_ranges(tmp_path, sample_video, monkeypatch):
    # Force a tiny cap so we can exercise the merge path without building
    # 200+ real ranges against a 10s source.
    monkeypatch.setattr(cutter, "MAX_KEEP_RANGES", 3)
    ranges = [(0.0, 0.5), (0.6, 1.1), (1.2, 1.7), (1.8, 2.3), (2.4, 2.9)]
    out_path = tmp_path / "capped.mp4"
    cutter.cut_ranges(sample_video, ranges, out_path)
    assert out_path.exists()
    # merge_small_gaps preserves total kept duration -- only touching/adjacent
    # gaps get folded together, no content is dropped.
    total = sum(e - s for s, e in ranges)
    assert _probe_duration(out_path) == pytest.approx(total, abs=0.4)


def test_cut_clip_still_works_as_thin_single_range_delegator(tmp_path, sample_video):
    from pipeline.schemas import Clip

    clip = Clip(id="clip-01", start=1.0, end=4.0, hook="h", topic="t", score=0.5)
    out_path = cutter.cut_clip(sample_video, clip, tmp_path / "clips")
    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(3.0, abs=0.4)


def test_missing_source_raises_subprocess_failed_error(tmp_path):
    from pipeline import procutil

    with pytest.raises(procutil.SubprocessFailedError):
        cutter.cut_ranges(tmp_path / "does_not_exist.mp4", [(0.0, 1.0)], tmp_path / "out.mp4")
