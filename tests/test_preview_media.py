"""Real-ffmpeg tests for pipeline/preview.py's frame extraction against
synthetic lavfi media -- no network, no checked-in fixtures, matches
tests/test_media.py's pattern."""

import shutil
import subprocess

import pytest

from pipeline import preview

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("preview-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def test_extract_frame_produces_a_png(tmp_path, sample_video):
    out_path = tmp_path / "frame.png"
    result = preview.extract_frame(sample_video, 1.5, out_path)
    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_extract_frames_produces_one_file_per_timestamp(tmp_path, sample_video):
    paths = preview.extract_frames(sample_video, [0.0, 1.0, 2.0], tmp_path / "frames")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)


def test_extract_frame_creates_parent_dirs(tmp_path, sample_video):
    out_path = tmp_path / "nested" / "dir" / "frame.png"
    preview.extract_frame(sample_video, 0.5, out_path)
    assert out_path.exists()


def test_extract_frame_missing_source_raises(tmp_path):
    from pipeline import procutil

    with pytest.raises(procutil.SubprocessFailedError):
        preview.extract_frame(tmp_path / "does_not_exist.mp4", 0.0, tmp_path / "out.png")
