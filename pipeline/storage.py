"""Run directory conventions and JSON I/O helpers.

Every run lives under ``<data_dir>/<run_id>/`` with one JSON file per stage
(``transcript.json``, ``clips.json``, ``posts.json``, ...). This is the
on-disk form of the "deterministic JSON between stages" contract -- any
orchestrator (this CLI today, Pipedream/Activepieces later) just needs to
read/write these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from pipeline.config import get_settings
from pipeline.schemas import PipelineRun

T = TypeVar("T", bound=BaseModel)


def run_dir(run_id: str) -> Path:
    return get_settings().data_dir / run_id


def stage_path(run_id: str, stage: str) -> Path:
    """Path for a stage's JSON artifact, e.g. stage='transcript' -> transcript.json."""
    return run_dir(run_id) / f"{stage}.json"


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def read_json(path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def write_json_list(path: Path, models: list[BaseModel]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [m.model_dump(mode="json") for m in models]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json_list(path: Path, model_cls: type[T]) -> list[T]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [model_cls.model_validate(item) for item in raw]


def init_run(source: str, run_id: str) -> PipelineRun:
    run = PipelineRun(run_id=run_id, source=source, status="in_progress")
    save_run_state(run)
    return run


def run_state_path(run_id: str) -> Path:
    return stage_path(run_id, "run")


def save_run_state(run: PipelineRun) -> None:
    write_json(run_state_path(run.run_id), run)


def load_run_state(run_id: str) -> PipelineRun:
    return read_json(run_state_path(run_id), PipelineRun)
