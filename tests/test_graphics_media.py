"""Real end-to-end HyperFrames test: renders an actual catalog composition
through the real `npx hyperframes render` and composites it onto a
synthetic clip -- gated on graphics_available() so it skips (never fails)
anywhere graphics/node_modules or Node itself isn't present. This box has
both (verified during Step 8's implementation: `npm install` inside
graphics/ succeeds against the real npm registry, and a pre-installed
Chrome-headless-shell binary lets `hyperframes render` run without any
further download), so this test exercises the real, non-stubbed path."""

import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline import graphics
from pipeline.schemas import Word

_GRAPHICS_DIR = Path(__file__).resolve().parent.parent / "graphics"
_AVAILABLE, _REASON = graphics.graphics_available(_GRAPHICS_DIR)

pytestmark = [
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
    pytest.mark.skipif(not _AVAILABLE, reason=f"graphics stage not available: {_REASON}"),
]


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("gfx-media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=640x360:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def test_render_beat_produces_alpha_mov(tmp_path):
    catalog = graphics.load_catalog(_GRAPHICS_DIR)
    spec = catalog["stat-card"]
    from pipeline.schemas import GraphicsBeat

    beat = GraphicsBeat(composition="stat-card", variables={"value": "3.2x", "label": "growth"}, start=0.0, duration=spec.duration)

    out_path = graphics._render_beat(beat, spec, _GRAPHICS_DIR, tmp_path / "renders", fps=24, timeout=90)

    assert out_path.exists()
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    assert "prores" in proc.stdout
    assert "yuva" in proc.stdout  # has an alpha channel


def test_render_beat_is_cached_on_identical_inputs(tmp_path):
    catalog = graphics.load_catalog(_GRAPHICS_DIR)
    spec = catalog["keyword-pop"]
    from pipeline.schemas import GraphicsBeat

    beat = GraphicsBeat(composition="keyword-pop", variables={"word": "NEVER"}, start=0.0, duration=spec.duration)
    cache_dir = tmp_path / "renders"

    first = graphics._render_beat(beat, spec, _GRAPHICS_DIR, cache_dir, fps=24, timeout=90)
    mtime_after_first = first.stat().st_mtime

    second = graphics._render_beat(beat, spec, _GRAPHICS_DIR, cache_dir, fps=24, timeout=90)
    assert second == first
    assert second.stat().st_mtime == mtime_after_first  # not re-rendered


def test_plan_and_render_graphics_full_pipeline(tmp_path, sample_video, monkeypatch):
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [{
                "composition": "stat-card", "variables": {"value": "3.2x", "label": "growth"},
                "anchor_word": "grew three", "confidence": 0.9,
            }]}

    monkeypatch.setattr(graphics, "get_llm_client", lambda: FakeClient())

    words = [
        Word(text="revenue", start=0.0, end=0.4),
        Word(text="grew", start=0.5, end=0.8),
        Word(text="three", start=0.9, end=1.1),
        Word(text="x.", start=1.1, end=1.3),
    ]
    out_path = tmp_path / "composited.mp4"

    plan = graphics.plan_and_render_graphics(
        "clip-01", sample_video, words, out_path, graphics_dir=_GRAPHICS_DIR, fps=24, render_timeout_s=90,
    )

    assert plan.method == "llm"
    assert len(plan.beats) == 1
    assert out_path.exists()
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    assert float(proc.stdout.strip()) == pytest.approx(4.0, abs=0.3)  # base duration preserved
