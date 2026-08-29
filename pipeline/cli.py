"""Typer CLI: one subcommand per pipeline stage, plus a composite `run`.

Every stage reads/writes the JSON artifacts under runs/<run_id>/ (see
pipeline/storage.py), so each subcommand is independently runnable --
useful for debugging a single stage, and for resuming `run` after a stage
fails partway (e.g. a stalled download over flaky connectivity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pipeline import analyst, captioner, clip_selector, cutter, ingest, publisher, repurposer, transcribe as transcribe_
from pipeline.config import new_run_id
from pipeline.schemas import (
    Clip,
    DEFAULT_PLATFORMS,
    MediaAsset,
    PerformanceRecord,
    Platform,
    Post,
    PublishResult,
    Transcript,
)
from pipeline.storage import (
    init_run,
    load_run_state,
    read_json,
    read_json_list,
    run_dir,
    save_run_state,
    stage_path,
    write_json,
    write_json_list,
)

app = typer.Typer(add_completion=False, help="Zero-key content repurposing pipeline.")
console = Console()


# --------------------------------------------------------------------------
# Path-map helpers (clip_id -> media file path), used for raw/captioned clips
# --------------------------------------------------------------------------

def _save_path_map(path: Path, mapping: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: str(v) for k, v in mapping.items()}, indent=2), encoding="utf-8")


def _load_path_map(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: Path(v) for k, v in raw.items()}


def _parse_platforms(platforms: str) -> tuple[Platform, ...]:
    names = [p.strip() for p in platforms.split(",") if p.strip()]
    return tuple(names) if names else DEFAULT_PLATFORMS  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Per-stage subcommands
# --------------------------------------------------------------------------

@app.command("ingest")
def ingest_cmd(source: str, run_id: Optional[str] = None) -> str:
    run_id = run_id or new_run_id()
    run = init_run(source, run_id)
    console.print(f"[bold]ingest[/bold] -> run_id={run_id}")

    asset = ingest.ingest(source, run_id)
    write_json(stage_path(run_id, "media"), asset)

    run.mark_done("ingest")
    save_run_state(run)
    console.print(f"  source: {asset.local_path} ({asset.duration:.1f}s)")
    return run_id


@app.command("transcribe")
def transcribe_cmd(run_id: str) -> None:
    asset = read_json(stage_path(run_id, "media"), MediaAsset)
    console.print(f"[bold]transcribe[/bold] run_id={run_id} (this downloads the whisper model on first use)")

    transcript = transcribe_.transcribe(Path(asset.local_path), run_id)
    write_json(stage_path(run_id, "transcript"), transcript)

    run = load_run_state(run_id)
    run.mark_done("transcribe")
    save_run_state(run)
    console.print(f"  {len(transcript.segments)} segments, {transcript.duration:.1f}s, language={transcript.language}")


@app.command("select-clips")
def select_clips_cmd(
    run_id: str,
    max_clips: int = clip_selector.DEFAULT_MAX_CLIPS,
    min_len: float = clip_selector.DEFAULT_MIN_LEN,
    max_len: float = clip_selector.DEFAULT_MAX_LEN,
) -> None:
    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    learnings = analyst.load_learnings()

    clips = clip_selector.select_clips(transcript, max_clips=max_clips, min_len=min_len, max_len=max_len, learnings=learnings)
    write_json_list(stage_path(run_id, "clips"), clips)

    run = load_run_state(run_id)
    run.mark_done("select_clips")
    save_run_state(run)

    table = Table(title=f"select-clips run_id={run_id}")
    table.add_column("id")
    table.add_column("start")
    table.add_column("end")
    table.add_column("score")
    table.add_column("hook")
    for c in clips:
        table.add_row(c.id, f"{c.start:.1f}", f"{c.end:.1f}", f"{c.score:.2f}", c.hook[:60])
    console.print(table)


@app.command("cut")
def cut_cmd(run_id: str) -> None:
    asset = read_json(stage_path(run_id, "media"), MediaAsset)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)

    raw = cutter.cut_all(Path(asset.local_path), clips, run_dir(run_id) / "clips_raw")
    _save_path_map(stage_path(run_id, "raw_clips"), raw)

    run = load_run_state(run_id)
    run.mark_done("cut")
    save_run_state(run)
    console.print(f"[bold]cut[/bold] run_id={run_id}: {len(raw)} clip(s) cut")


@app.command("caption")
def caption_cmd(run_id: str) -> None:
    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    raw = _load_path_map(stage_path(run_id, "raw_clips"))

    all_words = transcript.all_words()
    captioned: dict[str, Path] = {}
    for clip in clips:
        clip_words = [w for w in all_words if w.start >= clip.start and w.end <= clip.end]
        out_path = run_dir(run_id) / "clips_captioned" / f"{clip.id}.mp4"
        captioner.render_captioned_clip(raw[clip.id], clip_words, out_path, offset=clip.start)
        captioned[clip.id] = out_path

    _save_path_map(stage_path(run_id, "captioned_clips"), captioned)

    run = load_run_state(run_id)
    run.mark_done("caption")
    save_run_state(run)
    console.print(f"[bold]caption[/bold] run_id={run_id}: {len(captioned)} clip(s) captioned")


@app.command("repurpose")
def repurpose_cmd(run_id: str, platforms: str = ",".join(DEFAULT_PLATFORMS)) -> None:
    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    learnings = analyst.load_learnings()
    platform_tuple = _parse_platforms(platforms)

    posts: list[Post] = []
    for clip in clips:
        posts.extend(repurposer.generate_posts(clip, transcript, platforms=platform_tuple, learnings=learnings))
    write_json_list(stage_path(run_id, "posts"), posts)

    run = load_run_state(run_id)
    run.mark_done("repurpose")
    save_run_state(run)
    console.print(f"[bold]repurpose[/bold] run_id={run_id}: {len(posts)} post(s) across {len(clips)} clip(s)")


@app.command("publish")
def publish_cmd(run_id: str) -> None:
    posts = read_json_list(stage_path(run_id, "posts"), Post)
    captioned_path = stage_path(run_id, "captioned_clips")
    media = _load_path_map(captioned_path) if captioned_path.exists() else {}

    results = publisher.publish(run_id, posts, clip_media=media)
    write_json_list(stage_path(run_id, "publish_results"), results)

    run = load_run_state(run_id)
    run.mark_done("publish")
    save_run_state(run)

    method = results[0].method if results else "outbox"
    console.print(f"[bold]publish[/bold] run_id={run_id}: {len(results)} post(s) via {method}")
    if method == "outbox":
        console.print(f"  review at: {run_dir(run_id) / 'outbox'}")


@app.command("analyze")
def analyze_cmd(
    performance_file: Path,
    run_id: Optional[str] = typer.Option(None, help="Run to pull posts/clips from for keyword extraction"),
    learnings_path: Path = analyst.DEFAULT_LEARNINGS_PATH,
) -> None:
    records = analyst.load_performance(performance_file)
    posts = read_json_list(stage_path(run_id, "posts"), Post) if run_id else None
    clips = read_json_list(stage_path(run_id, "clips"), Clip) if run_id else None

    learnings = analyst.analyze(records, posts=posts, clips=clips)
    analyst.save_learnings(learnings, learnings_path)

    console.print(f"[bold]analyze[/bold]: {len(records)} record(s) -> {learnings_path}")
    console.print(f"  best platforms: {', '.join(learnings.best_platforms) or 'n/a'}")
    console.print(f"  top keywords: {', '.join(learnings.top_keywords) or 'n/a'}")


# --------------------------------------------------------------------------
# Composite, resumable full pipeline
# --------------------------------------------------------------------------

STAGES = ("ingest", "transcribe", "select_clips", "cut", "caption", "repurpose", "publish")


@app.command("run")
def run_cmd(
    source: str,
    run_id: Optional[str] = typer.Option(None, help="Resume this run_id instead of starting a new one"),
    max_clips: int = clip_selector.DEFAULT_MAX_CLIPS,
    min_len: float = clip_selector.DEFAULT_MIN_LEN,
    max_len: float = clip_selector.DEFAULT_MAX_LEN,
    platforms: str = ",".join(DEFAULT_PLATFORMS),
) -> None:
    """Runs every stage in order. Safe to re-run with the same --run-id after
    an interruption: completed stages are skipped."""
    if run_id and stage_path(run_id, "run").exists():
        run = load_run_state(run_id)
        console.print(f"[bold]resuming[/bold] run_id={run_id}, completed so far: {run.stages_completed}")
    else:
        run_id = ingest_cmd(source, run_id)
        run = load_run_state(run_id)

    if "transcribe" not in run.stages_completed:
        transcribe_cmd(run_id)
        run = load_run_state(run_id)
    if "select_clips" not in run.stages_completed:
        select_clips_cmd(run_id, max_clips=max_clips, min_len=min_len, max_len=max_len)
        run = load_run_state(run_id)
    if "cut" not in run.stages_completed:
        cut_cmd(run_id)
        run = load_run_state(run_id)
    if "caption" not in run.stages_completed:
        caption_cmd(run_id)
        run = load_run_state(run_id)
    if "repurpose" not in run.stages_completed:
        repurpose_cmd(run_id, platforms=platforms)
        run = load_run_state(run_id)
    if "publish" not in run.stages_completed:
        publish_cmd(run_id)
        run = load_run_state(run_id)

    run.status = "completed"
    save_run_state(run)
    console.print(f"[bold green]done[/bold green] run_id={run_id}")


if __name__ == "__main__":
    app()
