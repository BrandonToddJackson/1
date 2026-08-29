"""Typer CLI: one subcommand per pipeline stage, plus a composite `run`.

Every stage reads/writes the JSON artifacts under runs/<run_id>/ (see
pipeline/storage.py). Each stage is split into a plain `_stage_x(...)`
function (called both by its thin `@app.command()` wrapper and directly by
`run_cmd`'s loop -- calling a Typer-decorated command function directly
would pass `typer.OptionInfo` objects, not values, for any unset Option
default) plus the wrapper, which turns a PipelineError into a clean printed
message + exit code instead of a traceback.

Resumability model: a stage that overwrites its own artifact first
invalidates -- un-marks-done, deletes on-disk output for -- itself and
every LATER stage (see _invalidate_from), so a stale `cut`/`caption`/
`publish` output can never survive a `select-clips` re-run with different
parameters. The one deliberate exception is `cut`/`caption`: they do NOT
self-invalidate, so a retry after a partial failure resumes by skipping
clips whose output already exists rather than redoing the whole batch --
safe only because select-clips (the stage that actually changes their
input) always clears their output first. `run` persists the tuning params
it used on the run and refuses to silently reuse stale output if asked for
different ones without `--force`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pipeline import analyst, audio, captioner, clip_selector, cutter, ingest, publisher, repurposer
from pipeline import transcribe as transcribe_
from pipeline.config import get_settings, new_run_id
from pipeline.schemas import (
    Clip,
    DEFAULT_PLATFORMS,
    MediaAsset,
    PipelineRun,
    Platform,
    Post,
    Transcript,
)
from pipeline.storage import (
    init_run,
    load_run_state,
    read_json,
    read_json_list,
    remove_run_path,
    run_dir,
    save_run_state,
    stage_path,
    write_json,
    write_json_list,
)

app = typer.Typer(add_completion=False, help="Zero-key content repurposing pipeline.")
console = Console()
log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised by a _stage_* function: a missing prerequisite artifact, or
    the underlying stage logic failing. Caught by the thin @app.command()
    wrappers (-> printed message + typer.Exit(1)) and by run_cmd's loop
    (-> PipelineRun.mark_failed). Deliberately never typer.Exit itself
    (also a RuntimeError subclass) -- _stage_* functions never raise
    typer.Exit, so run_cmd's loop, which calls them directly, only ever
    sees PipelineError from a real stage problem."""


STAGE_ORDER: tuple[str, ...] = (
    "ingest", "enhance", "transcribe", "select_clips", "cut", "caption", "repurpose", "publish",
)

# Artifacts (glob patterns relative to the run directory) each stage owns --
# deleted whenever that stage or an earlier one is invalidated, so stale
# output can never be mistaken for current.
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "ingest": ("media.json", "source.*"),
    "enhance": ("enhanced_media.json", "enhanced.*"),
    "transcribe": ("transcript.json",),
    "select_clips": ("clips.json",),
    "cut": ("raw_clips.json", "clips_raw"),
    "caption": ("captioned_clips.json", "clips_captioned"),
    "repurpose": ("posts.json",),
    "publish": ("publish_results.json", "outbox"),
}

# Artifacts each stage requires from an earlier stage before it can run.
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "enhance": ("media.json",),
    "transcribe": ("enhanced_media.json",),
    "select_clips": ("transcript.json",),
    "cut": ("media.json", "clips.json"),
    "caption": ("transcript.json", "clips.json", "raw_clips.json"),
    "repurpose": ("transcript.json", "clips.json"),
    "publish": ("posts.json",),
}

_PRODUCED_BY: dict[str, str] = {
    "media.json": "ingest",
    "enhanced_media.json": "enhance",
    "transcript.json": "transcribe",
    "clips.json": "select-clips",
    "raw_clips.json": "cut",
    "posts.json": "repurpose",
}

# Which stage a `run` tuning param first affects -- a changed param
# invalidates from here onward (with --force), not the whole pipeline.
PARAM_AFFECTS: dict[str, str] = {
    "source": "ingest",
    "enhance": "enhance",
    "max_clips": "select_clips",
    "min_len": "select_clips",
    "max_len": "select_clips",
    "platforms": "repurpose",
}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _artifact_path(run_id: str, name: str) -> Path:
    return run_dir(run_id) / name


def _require_stage_artifact(run_id: str, stage: str) -> None:
    for artifact in STAGE_REQUIRES.get(stage, ()):
        if not _artifact_path(run_id, artifact).exists():
            producer = _PRODUCED_BY.get(artifact, "an earlier stage")
            raise PipelineError(f"{stage}: missing {artifact} -- run `{producer}` first (or `run`)")


def _invalidate_from(run_id: str, run: PipelineRun, stage: str) -> None:
    """Un-marks `stage` and every later stage as completed, and deletes
    their on-disk artifacts. Called at the START of a stage (before any new
    work), so a crash mid-stage never leaves a later stage's stale output
    looking current. Mutates `run` in place and saves it."""
    idx = STAGE_ORDER.index(stage)
    base = run_dir(run_id)
    for later_stage in STAGE_ORDER[idx:]:
        run.undo(later_stage)
        for pattern in STAGE_ARTIFACTS.get(later_stage, ()):
            if base.exists():
                for p in base.glob(pattern):
                    remove_run_path(run_id, str(p.relative_to(base)))
    save_run_state(run)


def _save_path_map(path: Path, mapping: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: str(v) for k, v in mapping.items()}, indent=2), encoding="utf-8")


def _load_path_map(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: Path(v) for k, v in raw.items()}


def _parse_platforms(platforms: str) -> tuple[Platform, ...]:
    names = [p.strip() for p in platforms.split(",") if p.strip()]
    if not names:
        return DEFAULT_PLATFORMS
    invalid = [p for p in names if p not in repurposer.PLATFORM_RULES]
    if invalid:
        valid = ", ".join(repurposer.PLATFORM_RULES.keys())
        raise typer.BadParameter(f"unknown platform(s): {', '.join(invalid)}; valid: {valid}")
    return tuple(names)  # type: ignore[return-value]


def _fail(stage_label: str, exc: BaseException) -> None:
    console.print(f"[bold red]failed[/bold red] ({stage_label}): {exc}")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# Plain stage functions (no Typer defaults -- safe to call directly)
# --------------------------------------------------------------------------

def _stage_ingest(source: str, run_id: str) -> None:
    if stage_path(run_id, "run").exists():
        run = load_run_state(run_id)
    else:
        run = init_run(source, run_id)
    _invalidate_from(run_id, run, "ingest")
    console.print(f"[bold]ingest[/bold] -> run_id={run_id}")

    try:
        asset = ingest.ingest(source, run_id)
    except Exception as exc:  # noqa: BLE001 - normalize to PipelineError for uniform handling
        raise PipelineError(f"ingest: {exc}") from exc
    write_json(stage_path(run_id, "media"), asset)

    run = load_run_state(run_id)
    run.source = source
    run.params["source"] = source
    run.mark_done("ingest")
    save_run_state(run)
    console.print(f"  source: {asset.local_path} ({asset.duration:.1f}s)")


def _stage_enhance(run_id: str, enabled: bool = True) -> None:
    """Denoise + normalize the source audio once, before transcribe (a
    per-clip loudnorm would give every clip a different perceived volume,
    and normalized/denoised audio measurably improves whisper accuracy for
    everything downstream). Disabled is an identity artifact, not a missing
    one -- enhanced_media.json always exists, pointing at the source as-is
    when --no-enhance is passed, so `transcribe` never needs to branch on
    whether enhancement ran."""
    _require_stage_artifact(run_id, "enhance")
    run = load_run_state(run_id)
    _invalidate_from(run_id, run, "enhance")

    asset = read_json(stage_path(run_id, "media"), MediaAsset)
    settings = get_settings()

    if not enabled:
        console.print(f"[bold]enhance[/bold] run_id={run_id} (disabled, using source as-is)")
        enhanced = asset.model_copy(update={"enhanced_from": None, "loudness_lufs": None})
        write_json(stage_path(run_id, "enhanced_media"), enhanced)

        run = load_run_state(run_id)
        run.params["enhance"] = False
        run.mark_done("enhance")
        save_run_state(run)
        return

    console.print(f"[bold]enhance[/bold] run_id={run_id} (denoise + normalize)")
    src_path = Path(asset.local_path)
    out_path = run_dir(run_id) / f"enhanced{src_path.suffix or '.mp4'}"
    try:
        loudness = audio.enhance(
            src_path, out_path,
            target_lufs=settings.audio_target_lufs,
            rnnoise_model_path=settings.rnnoise_model_path,
        )
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"enhance: {exc}") from exc

    enhanced = asset.model_copy(update={
        "local_path": str(out_path),
        "enhanced_from": asset.local_path,
        "loudness_lufs": loudness,
    })
    write_json(stage_path(run_id, "enhanced_media"), enhanced)

    run = load_run_state(run_id)
    run.params["enhance"] = True
    run.mark_done("enhance")
    save_run_state(run)
    lufs_str = f"{loudness:.1f} LUFS" if loudness is not None else "unknown"
    console.print(f"  {out_path.name} ({lufs_str})")


def _stage_transcribe(run_id: str) -> None:
    _require_stage_artifact(run_id, "transcribe")
    run = load_run_state(run_id)
    _invalidate_from(run_id, run, "transcribe")

    asset = read_json(stage_path(run_id, "enhanced_media"), MediaAsset)
    console.print(f"[bold]transcribe[/bold] run_id={run_id} (this downloads the whisper model on first use)")

    try:
        transcript = transcribe_.transcribe(Path(asset.local_path), run_id)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"transcribe: {exc}") from exc
    write_json(stage_path(run_id, "transcript"), transcript)

    run = load_run_state(run_id)
    run.mark_done("transcribe")
    save_run_state(run)
    console.print(f"  {len(transcript.segments)} segments, {transcript.duration:.1f}s, language={transcript.language}")


def _stage_select_clips(run_id: str, max_clips: int, min_len: float, max_len: float) -> None:
    _require_stage_artifact(run_id, "select_clips")
    run = load_run_state(run_id)
    _invalidate_from(run_id, run, "select_clips")

    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    learnings = analyst.load_learnings()

    console.print(f"[bold]select-clips[/bold] run_id={run_id} (scoring candidate windows...)")
    try:
        clips = clip_selector.select_clips(transcript, max_clips=max_clips, min_len=min_len, max_len=max_len, learnings=learnings)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"select_clips: {exc}") from exc
    write_json_list(stage_path(run_id, "clips"), clips)

    run = load_run_state(run_id)
    run.params.update({"max_clips": max_clips, "min_len": min_len, "max_len": max_len})
    run.mark_done("select_clips")
    save_run_state(run)

    if not clips:
        console.print(
            "[bold yellow]warning[/bold yellow]: no clips selected -- try a lower --min-len, "
            "or check the transcript actually has enough speech."
        )
        return

    table = Table(title=f"select-clips run_id={run_id}")
    table.add_column("id")
    table.add_column("start")
    table.add_column("end")
    table.add_column("score")
    table.add_column("hook")
    for c in clips:
        table.add_row(c.id, f"{c.start:.1f}", f"{c.end:.1f}", f"{c.score:.2f}", c.hook[:60])
    console.print(table)


def _stage_cut(run_id: str) -> None:
    """Deliberately does NOT self-invalidate at the top (unlike every other
    stage) -- a retry after a partial ffmpeg failure resumes by skipping
    clips whose output file already exists, rather than redoing the whole
    batch. Safe only because select-clips invalidates (deletes) this
    stage's output whenever clips.json actually changes."""
    _require_stage_artifact(run_id, "cut")

    asset = read_json(stage_path(run_id, "media"), MediaAsset)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    out_dir = run_dir(run_id) / "clips_raw"
    raw_path = stage_path(run_id, "raw_clips")
    raw: dict[str, Path] = _load_path_map(raw_path) if raw_path.exists() else {}
    _save_path_map(raw_path, raw)  # ensure the file exists even with 0 clips (or 0 progress so far)

    failures: list[str] = []
    for clip in clips:
        existing = out_dir / f"{clip.id}.mp4"
        if existing.exists():
            raw[clip.id] = existing
            continue
        try:
            raw[clip.id] = cutter.cut_clip(Path(asset.local_path), clip, out_dir)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{clip.id}: {exc}")
            continue
        _save_path_map(raw_path, raw)  # persist after EACH success, not just at the end

    if failures:
        raise PipelineError("cut: " + "; ".join(failures))

    run = load_run_state(run_id)
    run.mark_done("cut")
    save_run_state(run)
    console.print(f"[bold]cut[/bold] run_id={run_id}: {len(raw)} clip(s) cut")


def _stage_caption(run_id: str) -> None:
    """Same resume-by-skipping-existing-output design as _stage_cut, and
    for the same reason -- see its docstring."""
    _require_stage_artifact(run_id, "caption")

    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    raw = _load_path_map(stage_path(run_id, "raw_clips"))
    all_words = transcript.all_words()

    out_dir = run_dir(run_id) / "clips_captioned"
    captioned_path = stage_path(run_id, "captioned_clips")
    captioned: dict[str, Path] = _load_path_map(captioned_path) if captioned_path.exists() else {}
    _save_path_map(captioned_path, captioned)  # ensure the file exists even with 0 clips

    failures: list[str] = []
    for clip in clips:
        out_path = out_dir / f"{clip.id}.mp4"
        if out_path.exists():
            captioned[clip.id] = out_path
            continue
        if clip.id not in raw:
            failures.append(f"{clip.id}: no cut clip found for this id (raw_clips.json out of sync -- re-run `cut`)")
            continue
        clip_words = [w for w in all_words if w.start >= clip.start and w.end <= clip.end]
        try:
            captioner.render_captioned_clip(raw[clip.id], clip_words, out_path, offset=clip.start)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{clip.id}: {exc}")
            continue
        captioned[clip.id] = out_path
        _save_path_map(captioned_path, captioned)  # persist after EACH success

    if failures:
        raise PipelineError("caption: " + "; ".join(failures))

    run = load_run_state(run_id)
    run.mark_done("caption")
    save_run_state(run)
    console.print(f"[bold]caption[/bold] run_id={run_id}: {len(captioned)} clip(s) captioned")


def _stage_repurpose(run_id: str, platforms: tuple[Platform, ...]) -> None:
    _require_stage_artifact(run_id, "repurpose")
    run = load_run_state(run_id)
    _invalidate_from(run_id, run, "repurpose")

    transcript = read_json(stage_path(run_id, "transcript"), Transcript)
    clips = read_json_list(stage_path(run_id, "clips"), Clip)
    learnings = analyst.load_learnings()

    try:
        posts: list[Post] = []
        for clip in clips:
            posts.extend(repurposer.generate_posts(clip, transcript, platforms=platforms, learnings=learnings))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"repurpose: {exc}") from exc
    write_json_list(stage_path(run_id, "posts"), posts)

    run = load_run_state(run_id)
    run.params["platforms"] = list(platforms)
    run.mark_done("repurpose")
    save_run_state(run)
    console.print(f"[bold]repurpose[/bold] run_id={run_id}: {len(posts)} post(s) across {len(clips)} clip(s)")


def _stage_publish(run_id: str) -> None:
    _require_stage_artifact(run_id, "publish")
    run = load_run_state(run_id)
    _invalidate_from(run_id, run, "publish")

    posts = read_json_list(stage_path(run_id, "posts"), Post)
    captioned_path = stage_path(run_id, "captioned_clips")
    media = _load_path_map(captioned_path) if captioned_path.exists() else {}

    try:
        results = publisher.publish(run_id, posts, clip_media=media)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"publish: {exc}") from exc
    write_json_list(stage_path(run_id, "publish_results"), results)

    run = load_run_state(run_id)
    run.mark_done("publish")
    save_run_state(run)

    method = results[0].method if results else "outbox"
    console.print(f"[bold]publish[/bold] run_id={run_id}: {len(results)} post(s) via {method}")
    if any(r.method == "outbox" for r in results):
        console.print(f"  review at: {run_dir(run_id) / 'outbox'}")


# --------------------------------------------------------------------------
# Thin @app.command() wrappers
# --------------------------------------------------------------------------

@app.command("ingest")
def ingest_cmd(source: str, run_id: Optional[str] = None) -> None:
    run_id = run_id or new_run_id()
    try:
        _stage_ingest(source, run_id)
    except PipelineError as exc:
        _fail("ingest", exc)
    console.print(f"run_id={run_id}")


@app.command("enhance")
def enhance_cmd(
    run_id: str,
    off: bool = typer.Option(False, "--off", help="Skip enhancement, use the source audio as-is"),
) -> None:
    try:
        _stage_enhance(run_id, enabled=not off)
    except PipelineError as exc:
        _fail("enhance", exc)


@app.command("transcribe")
def transcribe_cmd(run_id: str) -> None:
    try:
        _stage_transcribe(run_id)
    except PipelineError as exc:
        _fail("transcribe", exc)


@app.command("select-clips")
def select_clips_cmd(
    run_id: str,
    max_clips: int = clip_selector.DEFAULT_MAX_CLIPS,
    min_len: float = clip_selector.DEFAULT_MIN_LEN,
    max_len: float = clip_selector.DEFAULT_MAX_LEN,
) -> None:
    try:
        _stage_select_clips(run_id, max_clips, min_len, max_len)
    except PipelineError as exc:
        _fail("select-clips", exc)


@app.command("cut")
def cut_cmd(run_id: str) -> None:
    try:
        _stage_cut(run_id)
    except PipelineError as exc:
        _fail("cut", exc)


@app.command("caption")
def caption_cmd(run_id: str) -> None:
    try:
        _stage_caption(run_id)
    except PipelineError as exc:
        _fail("caption", exc)


@app.command("repurpose")
def repurpose_cmd(run_id: str, platforms: str = ",".join(DEFAULT_PLATFORMS)) -> None:
    platform_tuple = _parse_platforms(platforms)
    try:
        _stage_repurpose(run_id, platform_tuple)
    except PipelineError as exc:
        _fail("repurpose", exc)


@app.command("publish")
def publish_cmd(run_id: str) -> None:
    try:
        _stage_publish(run_id)
    except PipelineError as exc:
        _fail("publish", exc)


@app.command("analyze")
def analyze_cmd(
    performance_file: Path,
    run_id: Optional[str] = typer.Option(None, help="Run to pull posts/clips from for keyword extraction"),
    learnings_path: Path = analyst.DEFAULT_LEARNINGS_PATH,
) -> None:
    records = analyst.load_performance(performance_file)

    posts, clips = None, None
    if run_id:
        posts_path, clips_path = stage_path(run_id, "posts"), stage_path(run_id, "clips")
        if not posts_path.exists() or not clips_path.exists():
            console.print(
                f"[bold red]failed[/bold red]: run_id={run_id} is missing posts.json/clips.json "
                "-- has `repurpose` completed for this run?"
            )
            raise typer.Exit(code=1)
        posts = read_json_list(posts_path, Post)
        clips = read_json_list(clips_path, Clip)

    learnings = analyst.analyze(records, posts=posts, clips=clips)
    analyst.save_learnings(learnings, learnings_path)

    console.print(f"[bold]analyze[/bold]: {len(records)} record(s) -> {learnings_path}")
    console.print(f"  best platforms: {', '.join(learnings.best_platforms) or 'n/a'}")
    console.print(f"  top keywords: {', '.join(learnings.top_keywords) or 'n/a'}")
    if learnings.notes:
        console.print(f"  notes: {learnings.notes}")


# --------------------------------------------------------------------------
# Composite, resumable full pipeline
# --------------------------------------------------------------------------

@app.command("run")
def run_cmd(
    source: str,
    run_id: Optional[str] = typer.Option(None, help="Resume this run_id instead of starting a new one"),
    max_clips: int = clip_selector.DEFAULT_MAX_CLIPS,
    min_len: float = clip_selector.DEFAULT_MIN_LEN,
    max_len: float = clip_selector.DEFAULT_MAX_LEN,
    platforms: str = ",".join(DEFAULT_PLATFORMS),
    no_enhance: bool = typer.Option(False, "--no-enhance", help="Skip audio enhancement, use the source as-is"),
    force: bool = typer.Option(False, "--force", help="Redo from the earliest stage affected by changed params"),
) -> None:
    """Runs every stage in order. Safe to re-run with the same --run-id:
    already-completed stages are skipped if params are unchanged. If you
    pass different tuning params than a previous run of this run_id used,
    this refuses to silently reuse stale output -- pass --force to redo
    from the earliest affected stage onward."""
    platform_tuple = _parse_platforms(platforms)  # fail fast on a typo before any stage (or run dir) exists
    requested_params = {
        "source": source,
        "enhance": not no_enhance,
        "max_clips": max_clips,
        "min_len": min_len,
        "max_len": max_len,
        "platforms": list(platform_tuple),
    }

    run_id = run_id or new_run_id()
    if stage_path(run_id, "run").exists():
        run = load_run_state(run_id)
        console.print(f"[bold]resuming[/bold] run_id={run_id}, completed so far: {run.stages_completed}")

        changed = sorted(k for k, v in requested_params.items() if run.params.get(k, v) != v)
        if changed:
            if not force:
                console.print(
                    f"[bold red]refusing[/bold red]: run_id={run_id} was previously run with different "
                    f"params ({', '.join(changed)} changed). Pass --force to redo from the earliest "
                    "affected stage, or omit the changed options to resume as-is."
                )
                raise typer.Exit(code=1)
            affected = [PARAM_AFFECTS[k] for k in changed if k in PARAM_AFFECTS]
            if affected:
                earliest = min(affected, key=STAGE_ORDER.index)
                console.print(f"[bold]--force[/bold]: redoing from '{earliest}' onward ({', '.join(changed)} changed)")
                _invalidate_from(run_id, run, earliest)
    else:
        run = init_run(source, run_id)

    if set(STAGE_ORDER).issubset(run.stages_completed):
        console.print(f"[bold]nothing to do[/bold]: all stages already complete for run_id={run_id}")
        return

    stage_fns = {
        "ingest": lambda: _stage_ingest(source, run_id),
        "enhance": lambda: _stage_enhance(run_id, enabled=not no_enhance),
        "transcribe": lambda: _stage_transcribe(run_id),
        "select_clips": lambda: _stage_select_clips(run_id, max_clips, min_len, max_len),
        "cut": lambda: _stage_cut(run_id),
        "caption": lambda: _stage_caption(run_id),
        "repurpose": lambda: _stage_repurpose(run_id, platform_tuple),
        "publish": lambda: _stage_publish(run_id),
    }

    for stage in STAGE_ORDER:
        if stage in run.stages_completed:
            continue
        try:
            stage_fns[stage]()
        except PipelineError as exc:
            run = load_run_state(run_id) if stage_path(run_id, "run").exists() else run
            run.mark_failed(stage, exc)
            save_run_state(run)
            console.print(f"[bold red]failed[/bold red] at stage '{stage}': {exc}")
            raise typer.Exit(code=1)
        run = load_run_state(run_id)

    run.status = "completed"
    save_run_state(run)

    clips_path = stage_path(run_id, "clips")
    clip_count = len(read_json_list(clips_path, Clip)) if clips_path.exists() else 0
    if clip_count == 0:
        console.print(f"[bold green]done[/bold green] run_id={run_id} (0 clips selected -- nothing to cut/publish)")
    else:
        console.print(f"[bold green]done[/bold green] run_id={run_id}")


if __name__ == "__main__":
    app()
