"""Real-ffmpeg tests for graphics._composite() against a SELF-GENERATED
alpha MOV (qtrle codec) -- exercises the compositor's overlay/timing logic
without needing HyperFrames/Node installed at all, per the plan's
verification strategy. See test_graphics_media.py for a real end-to-end
HyperFrames render (gated on graphics_available())."""

import shutil
import subprocess

import pytest

from pipeline import graphics
from pipeline.schemas import GraphicsBeat

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def base_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("gfx-composite") / "base.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def _make_alpha_mov(out_path, duration=1.0, color="red@0.6"):
    """A tiny self-generated alpha-channel MOV (qtrle) -- stands in for a
    real HyperFrames render so the compositor can be tested without Node."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=size=160x120:color={color}:duration={duration}:rate=15",
        "-vf", "format=yuva420p",
        "-c:v", "qtrle",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


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


def test_composite_single_beat_preserves_base_duration(tmp_path, base_video):
    alpha = _make_alpha_mov(tmp_path / "beat.mov")
    beat = GraphicsBeat(composition="fake", variables={}, start=1.0, duration=1.0)
    out_path = tmp_path / "out.mp4"

    graphics._composite(base_video, [(beat, alpha)], out_path)

    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(_probe_duration(base_video), abs=0.3)


def test_composite_preserves_audio(tmp_path, base_video):
    alpha = _make_alpha_mov(tmp_path / "beat.mov")
    beat = GraphicsBeat(composition="fake", variables={}, start=1.0, duration=1.0)
    out_path = tmp_path / "out.mp4"

    graphics._composite(base_video, [(beat, alpha)], out_path)
    assert _has_audio(out_path)


def test_composite_multiple_beats_chains_overlays(tmp_path, base_video):
    alpha1 = _make_alpha_mov(tmp_path / "beat1.mov", color="red@0.6")
    alpha2 = _make_alpha_mov(tmp_path / "beat2.mov", color="blue@0.6")
    beat1 = GraphicsBeat(composition="fake1", variables={}, start=0.5, duration=1.0)
    beat2 = GraphicsBeat(composition="fake2", variables={}, start=3.0, duration=1.0)
    out_path = tmp_path / "out.mp4"

    graphics._composite(base_video, [(beat1, alpha1), (beat2, alpha2)], out_path)

    assert out_path.exists()
    assert _probe_duration(out_path) == pytest.approx(_probe_duration(base_video), abs=0.3)


def test_composite_no_audio_source_still_succeeds(tmp_path, tmp_path_factory):
    silent = tmp_path_factory.mktemp("gfx-silent") / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=15", str(silent)],
        check=True, capture_output=True, text=True,
    )
    alpha = _make_alpha_mov(tmp_path / "beat.mov")
    beat = GraphicsBeat(composition="fake", variables={}, start=0.5, duration=1.0)
    out_path = tmp_path / "out.mp4"

    graphics._composite(silent, [(beat, alpha)], out_path)  # -map 0:a? must not fail here
    assert out_path.exists()


def test_composite_creates_parent_dirs(tmp_path, base_video):
    alpha = _make_alpha_mov(tmp_path / "beat.mov")
    beat = GraphicsBeat(composition="fake", variables={}, start=0.5, duration=1.0)
    out_path = tmp_path / "nested" / "dir" / "out.mp4"

    graphics._composite(base_video, [(beat, alpha)], out_path)
    assert out_path.exists()
