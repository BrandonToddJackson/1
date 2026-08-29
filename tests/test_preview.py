"""Pure-Python tests for pipeline/preview.py::default_preview_timestamps.
Real-ffmpeg frame-extraction tests live in test_preview_media.py."""

import pytest

from pipeline import preview
from pipeline.schemas import GraphicsBeat, GraphicsPlan


def test_default_timestamps_evenly_spaced_no_graphics():
    ts = preview.default_preview_timestamps(10.0, graphics_plan=None, n_evenly_spaced=5)
    assert len(ts) == 5
    assert ts[0] == 0.0
    assert ts[-1] == pytest.approx(10.0, abs=0.05)


def test_default_timestamps_includes_beat_midpoints():
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="stat-card", variables={}, start=3.0, duration=2.0)],
        method="llm",
    )
    ts = preview.default_preview_timestamps(10.0, graphics_plan=plan, n_evenly_spaced=5)
    assert 4.0 in ts  # midpoint of [3.0, 5.0)


def test_default_timestamps_dedupes_and_sorts():
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="stat-card", variables={}, start=0.0, duration=0.0)],
        method="llm",
    )
    ts = preview.default_preview_timestamps(10.0, graphics_plan=plan, n_evenly_spaced=5)
    assert ts == sorted(set(ts))
    assert ts.count(0.0) == 1  # beat midpoint (0.0) coincides with the first evenly-spaced frame


def test_default_timestamps_zero_duration_returns_empty():
    assert preview.default_preview_timestamps(0.0) == []
    assert preview.default_preview_timestamps(-1.0) == []


def test_default_timestamps_ignores_out_of_range_beat_midpoints():
    plan = GraphicsPlan(
        clip_id="clip-01",
        beats=[GraphicsBeat(composition="stat-card", variables={}, start=100.0, duration=2.0)],
        method="llm",
    )
    ts = preview.default_preview_timestamps(10.0, graphics_plan=plan, n_evenly_spaced=5)
    assert all(0 <= t <= 10.0 for t in ts)


def test_default_timestamps_single_frame_requested():
    ts = preview.default_preview_timestamps(10.0, n_evenly_spaced=1)
    assert ts == [5.0]
