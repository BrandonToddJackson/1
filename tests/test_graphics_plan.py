"""Pure-Python tests for pipeline/graphics.py's planning/degrade logic:
graphics_available()'s checks, catalog loading, anchor-word snapping, the
LLM planner's validation/capping, and plan_and_render_graphics()'s skip
ladder -- all against a stubbed LLM client and monkeypatched render/
composite steps, no Node/HyperFrames/ffmpeg needed. Real-ffmpeg compositing
lives in test_graphics_composite.py; a real end-to-end HyperFrames render
(gated on graphics_available()) lives in test_graphics_media.py."""

import json

import pytest

from pipeline import graphics
from pipeline.schemas import GraphicsBeat, GraphicsPlan, Word


# ---------------------------------------------------------------------------
# graphics_available()
# ---------------------------------------------------------------------------

def _make_project(tmp_path, with_node_modules=True, with_catalog=True):
    project = tmp_path / "graphics"
    project.mkdir()
    if with_node_modules:
        (project / "node_modules").mkdir()
    if with_catalog:
        (project / "catalog.json").write_text(json.dumps({"compositions": []}))
    return project


def test_graphics_available_true_when_everything_present(tmp_path):
    project = _make_project(tmp_path)
    ok, reason = graphics.graphics_available(project)
    assert ok is True
    assert reason == ""


def test_graphics_available_false_when_node_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics.shutil, "which", lambda name: None)
    project = _make_project(tmp_path)
    ok, reason = graphics.graphics_available(project)
    assert ok is False
    assert "node" in reason.lower()


def test_graphics_available_false_when_node_too_old(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics.shutil, "which", lambda name: "/usr/bin/node")

    class FakeProc:
        stdout = "v16.20.0\n"

    monkeypatch.setattr(graphics.subprocess, "run", lambda *a, **k: FakeProc())
    project = _make_project(tmp_path)
    ok, reason = graphics.graphics_available(project)
    assert ok is False
    assert "16" in reason


def test_graphics_available_false_when_node_modules_missing(tmp_path):
    project = _make_project(tmp_path, with_node_modules=False)
    ok, reason = graphics.graphics_available(project)
    assert ok is False
    assert "node_modules" in reason


def test_graphics_available_false_when_catalog_missing(tmp_path):
    project = _make_project(tmp_path, with_catalog=False)
    ok, reason = graphics.graphics_available(project)
    assert ok is False
    assert "catalog.json" in reason


def test_graphics_available_never_raises_on_broken_node_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics.shutil, "which", lambda name: "/usr/bin/node")

    def boom(*a, **k):
        raise OSError("exploded")

    monkeypatch.setattr(graphics.subprocess, "run", boom)
    project = _make_project(tmp_path)
    ok, reason = graphics.graphics_available(project)
    assert ok is False
    assert reason


# ---------------------------------------------------------------------------
# load_catalog
# ---------------------------------------------------------------------------

def test_load_catalog_parses_entries(tmp_path):
    project = tmp_path / "graphics"
    project.mkdir()
    (project / "catalog.json").write_text(json.dumps({
        "compositions": [
            {"id": "stat-card", "file": "compositions/stat-card.html", "description": "d", "duration": 2.0,
             "variables": [{"id": "value", "type": "string", "label": "L"}]},
        ]
    }))
    catalog = graphics.load_catalog(project)
    assert set(catalog) == {"stat-card"}
    assert catalog["stat-card"].duration == 2.0
    assert catalog["stat-card"].variable_ids == {"value"}


def test_real_catalog_loads_and_has_expected_compositions():
    from pathlib import Path

    catalog = graphics.load_catalog(Path("graphics"))
    assert set(catalog) == {"lower-third", "stat-card", "quote-card", "keyword-pop", "facecam-frame", "progress-dots"}
    for spec in catalog.values():
        assert spec.duration > 0
        assert spec.variable_ids  # every real composition declares at least one variable


# ---------------------------------------------------------------------------
# _snap_anchor
# ---------------------------------------------------------------------------

def _words():
    return [
        Word(text="revenue", start=0.0, end=0.4),
        Word(text="grew", start=0.5, end=0.8),
        Word(text="three", start=0.9, end=1.1),
        Word(text="x.", start=1.1, end=1.3),
    ]


def test_snap_anchor_matches_multi_word_phrase():
    w = graphics._snap_anchor("grew three", _words())
    assert w is not None
    assert w.text == "grew"
    assert w.start == 0.5


def test_snap_anchor_falls_back_to_first_token():
    w = graphics._snap_anchor("revenue increased massively", _words())
    assert w is not None
    assert w.text == "revenue"


def test_snap_anchor_case_and_punctuation_insensitive():
    w = graphics._snap_anchor("THREE X", _words())
    assert w is not None
    assert w.text == "three"


def test_snap_anchor_no_match_returns_none():
    assert graphics._snap_anchor("nonexistent phrase", _words()) is None


def test_snap_anchor_empty_anchor_or_words_returns_none():
    assert graphics._snap_anchor("", _words()) is None
    assert graphics._snap_anchor("revenue", []) is None


# ---------------------------------------------------------------------------
# _plan_beats_llm
# ---------------------------------------------------------------------------

def _catalog():
    return graphics.load_catalog(__import__("pathlib").Path("graphics"))


def test_plan_beats_llm_parses_valid_beat():
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [{
                "composition": "stat-card", "variables": {"value": "3.2x", "label": "growth"},
                "anchor_word": "grew three", "reason": "emphasis", "confidence": 0.9,
            }]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)
    assert plan.method == "llm"
    assert len(plan.beats) == 1
    assert plan.beats[0].composition == "stat-card"
    assert plan.beats[0].start == 0.5
    assert plan.beats[0].duration == 2.0


def test_plan_beats_llm_drops_unknown_composition():
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [{"composition": "bogus-composition", "variables": {}, "anchor_word": "grew"}]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)
    assert plan.method == "skipped"


def test_plan_beats_llm_drops_missing_variables():
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [{"composition": "stat-card", "variables": {"value": "3.2x"}, "anchor_word": "grew"}]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)
    assert plan.method == "skipped"  # missing required "label" variable


def test_plan_beats_llm_drops_unmatched_anchor():
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [{
                "composition": "stat-card", "variables": {"value": "3.2x", "label": "growth"},
                "anchor_word": "nonexistent phrase here",
            }]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)
    assert plan.method == "skipped"


def test_plan_beats_llm_malformed_entries_skipped_not_fatal():
    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": ["not a dict", {"composition": "stat-card"}, {}]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)
    assert plan.method == "skipped"


def test_plan_beats_llm_missing_beats_key_raises_llm_response_error():
    from pipeline.llm import LLMResponseError

    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"not_beats": []}

    with pytest.raises(LLMResponseError):
        graphics._plan_beats_llm(FakeClient(), "clip-01", _words(), _catalog(), max_beats=4)


def test_plan_beats_llm_caps_to_max_beats_by_confidence():
    words = [Word(text=f"w{i}", start=float(i), end=float(i) + 0.3) for i in range(10)]

    class FakeClient:
        def complete_json(self, system, user, schema_hint):
            return {"beats": [
                {"composition": "keyword-pop", "variables": {"word": "A"}, "anchor_word": "w0", "confidence": 0.2},
                {"composition": "keyword-pop", "variables": {"word": "B"}, "anchor_word": "w2", "confidence": 0.9},
                {"composition": "keyword-pop", "variables": {"word": "C"}, "anchor_word": "w4", "confidence": 0.5},
            ]}

    plan = graphics._plan_beats_llm(FakeClient(), "clip-01", words, _catalog(), max_beats=2)
    assert len(plan.beats) == 2
    # Highest-confidence beats kept (B, C), re-sorted chronologically by start.
    assert [b.variables["word"] for b in plan.beats] == ["B", "C"]


def test_plan_beats_llm_empty_words_returns_skipped():
    plan = graphics._plan_beats_llm(object(), "clip-01", [], _catalog(), max_beats=4)
    assert plan.method == "skipped"


# ---------------------------------------------------------------------------
# _beat_cache_key -- deterministic
# ---------------------------------------------------------------------------

def test_beat_cache_key_deterministic_for_identical_inputs():
    beat = GraphicsBeat(composition="stat-card", variables={"value": "3.2x", "label": "growth"}, anchor_word="grew", start=1.0, duration=2.0)
    assert graphics._beat_cache_key(beat, fps=30) == graphics._beat_cache_key(beat, fps=30)


def test_beat_cache_key_differs_on_variables():
    beat1 = GraphicsBeat(composition="stat-card", variables={"value": "3.2x"}, start=1.0, duration=2.0)
    beat2 = GraphicsBeat(composition="stat-card", variables={"value": "4.0x"}, start=1.0, duration=2.0)
    assert graphics._beat_cache_key(beat1, fps=30) != graphics._beat_cache_key(beat2, fps=30)


def test_beat_cache_key_differs_on_fps():
    beat = GraphicsBeat(composition="stat-card", variables={"value": "3.2x"}, start=1.0, duration=2.0)
    assert graphics._beat_cache_key(beat, fps=30) != graphics._beat_cache_key(beat, fps=24)


# ---------------------------------------------------------------------------
# plan_and_render_graphics -- the skip ladder (monkeypatched, no subprocess)
# ---------------------------------------------------------------------------

def test_plan_and_render_skips_when_graphics_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (False, "no node"))
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [], out)
    assert plan.method == "skipped"
    assert plan.skipped_reason == "no node"
    assert out.read_bytes() == b"fake"


def test_plan_and_render_skips_when_no_llm_key(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (True, ""))
    monkeypatch.setattr(graphics, "get_llm_client", lambda: None)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [], out)
    assert plan.method == "skipped"
    assert "LLM key" in plan.skipped_reason


def test_plan_and_render_skips_on_llm_planning_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (True, ""))
    monkeypatch.setattr(graphics, "get_llm_client", lambda: object())

    def boom(client, clip_id, words, catalog, max_beats):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(graphics, "_plan_beats_llm", boom)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [Word(text="hi", start=0.0, end=0.3)], out)
    assert plan.method == "skipped"
    assert "LLM planning failed" in plan.skipped_reason
    assert out.read_bytes() == b"fake"


def test_plan_and_render_one_failed_beat_dropped_others_still_composite(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (True, ""))
    monkeypatch.setattr(graphics, "get_llm_client", lambda: object())

    good_beat = GraphicsBeat(composition="stat-card", variables={"value": "3.2x", "label": "growth"}, start=0.5, duration=2.0)
    bad_beat = GraphicsBeat(composition="quote-card", variables={"quote": "hi"}, start=3.0, duration=2.5)
    fake_plan = GraphicsPlan(clip_id="clip-01", beats=[good_beat, bad_beat], method="llm")
    monkeypatch.setattr(graphics, "_plan_beats_llm", lambda *a, **k: fake_plan)
    monkeypatch.setattr(graphics, "load_catalog", lambda graphics_dir: {
        "stat-card": graphics.CompositionSpec("stat-card", "compositions/stat-card.html", "d", 2.0, []),
        "quote-card": graphics.CompositionSpec("quote-card", "compositions/quote-card.html", "d", 2.5, []),
    })

    def fake_render(beat, spec, graphics_dir, cache_dir, npx_bin="npx", fps=30, timeout=600.0):
        if beat.composition == "quote-card":
            raise RuntimeError("render exploded")
        return tmp_path / "rendered.mov"

    composite_calls = []

    def fake_composite(clip_path, rendered_beats, out_path):
        composite_calls.append(rendered_beats)
        out_path.write_bytes(b"composited")
        return out_path

    monkeypatch.setattr(graphics, "_render_beat", fake_render)
    monkeypatch.setattr(graphics, "_composite", fake_composite)

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [Word(text="hi", start=0.0, end=0.3)], out)
    assert len(plan.beats) == 1
    assert plan.beats[0].composition == "stat-card"
    assert len(composite_calls) == 1
    assert out.read_bytes() == b"composited"


def test_plan_and_render_all_beats_fail_copies_through_no_composite(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (True, ""))
    monkeypatch.setattr(graphics, "get_llm_client", lambda: object())

    beat = GraphicsBeat(composition="stat-card", variables={"value": "3.2x", "label": "growth"}, start=0.5, duration=2.0)
    fake_plan = GraphicsPlan(clip_id="clip-01", beats=[beat], method="llm")
    monkeypatch.setattr(graphics, "_plan_beats_llm", lambda *a, **k: fake_plan)
    monkeypatch.setattr(graphics, "load_catalog", lambda graphics_dir: {
        "stat-card": graphics.CompositionSpec("stat-card", "compositions/stat-card.html", "d", 2.0, []),
    })

    def boom(beat, spec, graphics_dir, cache_dir, npx_bin="npx", fps=30, timeout=600.0):
        raise RuntimeError("render exploded")

    composite_called = []
    monkeypatch.setattr(graphics, "_render_beat", boom)
    monkeypatch.setattr(graphics, "_composite", lambda *a, **k: composite_called.append(True))

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [Word(text="hi", start=0.0, end=0.3)], out)
    assert plan.beats == []
    assert not composite_called
    assert out.read_bytes() == b"fake"


def test_plan_and_render_zero_beats_planned_copies_through(tmp_path, monkeypatch):
    monkeypatch.setattr(graphics, "graphics_available", lambda graphics_dir, npx_bin="npx": (True, ""))
    monkeypatch.setattr(graphics, "get_llm_client", lambda: object())
    monkeypatch.setattr(
        graphics, "_plan_beats_llm",
        lambda *a, **k: GraphicsPlan(clip_id="clip-01", method="skipped", skipped_reason="LLM proposed no usable beats"),
    )

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake")
    out = tmp_path / "out.mp4"

    plan = graphics.plan_and_render_graphics("clip-01", src, [Word(text="hi", start=0.0, end=0.3)], out)
    assert plan.method == "skipped"
    assert out.read_bytes() == b"fake"
