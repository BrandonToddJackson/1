"""Real-ffmpeg tests for pipeline/qc.py's probing/measuring helpers
against synthetic lavfi media -- no network, no checked-in fixtures,
matches tests/test_media.py's pattern."""

import shutil
import subprocess

import pytest

from pipeline import qc
from pipeline.schemas import Clip

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("qc-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="module")
def silent_gap_video(tmp_path_factory):
    """1s tone, 2s true silence, 1s tone -- exercises silencedetect for real."""
    out = tmp_path_factory.mktemp("qc-silence") / "gap.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15",
        "-filter_complex",
        "sine=frequency=440:duration=1[a1];anullsrc=duration=2[a2];sine=frequency=440:duration=1[a3];"
        "[a1][a2][a3]concat=n=3:v=0:a=1[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def test_probe_duration_returns_real_duration(sample_video):
    measured = qc._probe_duration(sample_video)
    assert measured == pytest.approx(4.0, abs=0.3)


def test_probe_duration_missing_file_returns_none(tmp_path):
    assert qc._probe_duration(tmp_path / "does_not_exist.mp4") is None


def test_measure_loudness_returns_a_float(sample_video):
    measured = qc._measure_loudness(sample_video)
    assert measured is not None
    assert isinstance(measured, float)


def test_check_duration_real_probe_matches_expected(sample_video):
    clip = Clip(id="clip-01", start=0.0, end=4.0, hook="h", topic="t", score=0.5)
    assert qc.check_duration(clip, sample_video) is None  # within tolerance


def test_check_silence_detects_real_silent_gap(silent_gap_video):
    clip = Clip(id="clip-01", start=0.0, end=4.0, hook="h", topic="t", score=0.5)
    findings = qc.check_silence(clip, silent_gap_video, threshold=0.7)
    assert len(findings) >= 1
    assert any(f.measured is not None and f.measured >= 1.5 for f in findings)
