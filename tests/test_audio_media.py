"""Real-ffmpeg tests for pipeline/audio.py against synthetic lavfi media --
no network, no checked-in fixtures, matches tests/test_media.py's pattern."""

import shutil
import subprocess

import pytest

from pipeline import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("audio-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def _probe_video_codec_and_frames(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,nb_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return proc.stdout.strip()


def test_enhance_produces_valid_output_file(tmp_path, sample_video):
    out_path = tmp_path / "enhanced.mp4"
    loudness = audio.enhance(sample_video, out_path, target_lufs=-16.0)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert loudness is not None  # the JSON stats block should always be present


def test_enhance_leaves_video_stream_untouched(tmp_path, sample_video):
    out_path = tmp_path / "enhanced.mp4"
    audio.enhance(sample_video, out_path, target_lufs=-16.0)

    original = _probe_video_codec_and_frames(sample_video)
    enhanced = _probe_video_codec_and_frames(out_path)
    assert original == enhanced  # -c:v copy: codec + frame count identical


def test_enhance_output_has_audio_stream(tmp_path, sample_video):
    out_path = tmp_path / "enhanced.mp4"
    audio.enhance(sample_video, out_path, target_lufs=-16.0)

    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    assert "audio" in proc.stdout


def test_enhance_creates_parent_dirs(tmp_path, sample_video):
    out_path = tmp_path / "nested" / "dir" / "enhanced.mp4"
    audio.enhance(sample_video, out_path, target_lufs=-16.0)
    assert out_path.exists()


def test_enhance_with_missing_source_raises_subprocess_error(tmp_path):
    from pipeline import procutil

    with pytest.raises(procutil.SubprocessFailedError):
        audio.enhance(tmp_path / "does_not_exist.mp4", tmp_path / "out.mp4")
