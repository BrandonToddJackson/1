import os

from pipeline import storage
from pipeline.schemas import Clip, PipelineRun


def test_run_dir_and_stage_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert storage.run_dir("r1") == tmp_path / "r1"
    assert storage.stage_path("r1", "transcript") == tmp_path / "r1" / "transcript.json"


def test_write_and_read_json(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run = PipelineRun(run_id="r1", source="video.mp4", status="pending")
    path = storage.stage_path("r1", "run")
    storage.write_json(path, run)

    assert path.exists()
    loaded = storage.read_json(path, PipelineRun)
    assert loaded == run


def test_write_and_read_json_list(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    clips = [
        Clip(id="c1", start=0.0, end=10.0, hook="a", topic="t", score=0.5),
        Clip(id="c2", start=10.0, end=20.0, hook="b", topic="t", score=0.7),
    ]
    path = storage.stage_path("r1", "clips")
    storage.write_json_list(path, clips)

    loaded = storage.read_json_list(path, Clip)
    assert loaded == clips


def test_init_run_and_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    run = storage.init_run("video.mp4", "r1")
    assert run.status == "in_progress"
    assert storage.run_state_path("r1").exists()

    run.mark_done("ingest")
    storage.save_run_state(run)

    loaded = storage.load_run_state("r1")
    assert loaded.stages_completed == ["ingest"]
    assert loaded.source == "video.mp4"
