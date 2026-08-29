"""Clip + word timestamps -> LLM-planned motion-graphics overlays, rendered
via HyperFrames and composited onto the clip (the `graphics` stage).

Skip-shaped everywhere, per the project's governing rule: no LLM key, or
Node/graphics/node_modules missing -> GraphicsPlan(method="skipped") and
out_path is a straight copy of the input clip. One beat's render fails ->
that beat is dropped, the rest still composite. Only the FINAL ffmpeg
composite failing is a real error -- everything upstream of it degrades to
"fewer/no beats", never crashes the run.

The LLM never emits HTML. It picks a composition id from a fixed,
hand-authored catalog (graphics/catalog.json) and fills that composition's
declared, typed variables -- validated and clamped exactly like every
other LLM output in this repo (clip selection, declutter removals, post
copy). Composition duration is a catalog property, not an LLM output.
`hyperframes render --strict-variables` gives a second, independent
enforcement layer at render time (a bad variable set fails that beat's
render, not the whole batch).

Timestamp placement happens here, in Python, not in HyperFrames: the LLM
returns an `anchor_word` (a short quote from the transcript, not a
timestamp -- it never sees exact floats to hallucinate), and _snap_anchor
finds the nearest matching Word.

Every subprocess call goes through pipeline.procutil.run_or_raise with a
timeout (a headless-Chrome render can hang). Rendering is deterministically
cached: a beat's output filename is sha256(composition + variables +
duration + fps), so identical inputs skip the render entirely -- the same
resume-by-skipping pattern pipeline.cli's _stage_cut/_stage_caption use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.llm import LLMClient, LLMResponseError, get_llm_client
from pipeline.procutil import run_or_raise
from pipeline.schemas import GraphicsBeat, GraphicsPlan, Word

log = logging.getLogger(__name__)

DEFAULT_GRAPHICS_DIR = Path("graphics")
MIN_NODE_MAJOR = 22
GRAPHICS_MAX_BEATS_DEFAULT = 4
DEFAULT_FPS = 30
DEFAULT_RENDER_TIMEOUT_S = 600.0

# A moment worth emphasizing at most once every this many seconds -- see
# graphics/MOTION.md's "max beats per 30s" pacing rule. Enforced by the LLM
# prompt; graphics_max_beats (the hard cap on total beats) is the code-side
# backstop.
MAX_BEATS_PER_WINDOW = 3
BEAT_WINDOW_S = 30.0


@dataclass
class CompositionSpec:
    id: str
    file: str
    description: str
    duration: float
    variables: list[dict]  # [{"id": str, "type": str, "label": str}, ...]

    @property
    def variable_ids(self) -> set[str]:
        return {v["id"] for v in self.variables}


def load_catalog(graphics_dir: Path) -> dict[str, CompositionSpec]:
    data = json.loads((graphics_dir / "catalog.json").read_text(encoding="utf-8"))
    specs: dict[str, CompositionSpec] = {}
    for entry in data.get("compositions", []):
        specs[entry["id"]] = CompositionSpec(
            id=entry["id"],
            file=entry["file"],
            description=entry.get("description", ""),
            duration=float(entry["duration"]),
            variables=entry.get("variables", []),
        )
    return specs


def graphics_available(graphics_dir: Path | None = None, npx_bin: str = "npx") -> tuple[bool, str]:
    """Never raises -- returns (False, reason) for anything short of a
    fully working local HyperFrames setup: Node present, Node >= 22 (the
    minimum HyperFrames-generation-era Node this was built/tested against),
    graphics/node_modules present (`npm install` has been run), and
    graphics/catalog.json present."""
    graphics_dir = graphics_dir or DEFAULT_GRAPHICS_DIR
    node = shutil.which("node")
    if node is None:
        return False, "node is not installed / not on PATH"

    try:
        proc = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
        major = int(proc.stdout.strip().lstrip("v").split(".")[0])
    except Exception:  # noqa: BLE001 - any probing failure means "not available", not a crash
        return False, "could not determine node version"
    if major < MIN_NODE_MAJOR:
        return False, f"node {proc.stdout.strip()} found, but {MIN_NODE_MAJOR}+ is required"

    if not (graphics_dir / "node_modules").is_dir():
        return False, f"{graphics_dir / 'node_modules'} not found -- run `npm install` inside {graphics_dir}"
    if not (graphics_dir / "catalog.json").exists():
        return False, f"{graphics_dir / 'catalog.json'} not found"

    return True, ""


def _normalize_token(text: str) -> str:
    return text.strip(" .,!?;:\"'()-…").lower()


def _snap_anchor(anchor_word: str, words: list[Word]) -> Word | None:
    """Finds the transcript Word matching the LLM's anchor_word quote: a
    contiguous phrase match first (2+ words), falling back to matching just
    its first token. Returns None -- the beat is dropped, never crashes the
    batch -- if nothing matches at all."""
    if not anchor_word or not words:
        return None
    anchor_tokens = re.findall(r"[a-zA-Z0-9']+", anchor_word.lower())
    if not anchor_tokens:
        return None
    word_tokens = [_normalize_token(w.text) for w in words]

    n = len(anchor_tokens)
    if n > 1:
        for i in range(len(words) - n + 1):
            if word_tokens[i : i + n] == anchor_tokens:
                return words[i]

    first = anchor_tokens[0]
    for w, tok in zip(words, word_tokens):
        if tok == first:
            return w
    return None


def _plan_beats_llm(
    client: LLMClient, clip_id: str, words: list[Word], catalog: dict[str, CompositionSpec], max_beats: int
) -> GraphicsPlan:
    if not words:
        return GraphicsPlan(clip_id=clip_id, method="skipped", skipped_reason="no words in this clip")

    catalog_desc = [
        {"id": spec.id, "description": spec.description, "variables": sorted(spec.variable_ids)}
        for spec in catalog.values()
    ]
    system = (
        "You plan motion-graphics overlays for a short video clip. Pick moments in the "
        "transcript worth visually emphasizing and choose ONE composition from the catalog "
        f"for each: {json.dumps(catalog_desc)}. Quote a short anchor_word phrase (2-5 exact "
        "words from the transcript) for each beat -- it will be snapped to that exact spot. "
        "Fill every variable the chosen composition declares (as strings). Use at most "
        f"{max_beats} beats total, and never more than {MAX_BEATS_PER_WINDOW} in any "
        f"{BEAT_WINDOW_S:.0f}-second window -- graphics are emphasis, not decoration; when in "
        "doubt, propose fewer, not more."
    )
    transcript_line = " ".join(w.text for w in words)
    schema_hint = (
        '{"beats": [{"composition": str, "variables": {"<id>": str, ...}, '
        '"anchor_word": str, "reason": str, "confidence": float (0-1)}]}'
    )
    result = client.complete_json(system=system, user=transcript_line, schema_hint=schema_hint)

    raw_beats = result.get("beats")
    if not isinstance(raw_beats, list):
        raise LLMResponseError("LLM graphics response missing a 'beats' list")

    candidates: list[tuple[float, GraphicsBeat]] = []
    for raw in raw_beats:
        if not isinstance(raw, dict):
            log.warning("LLM graphics beat entry is not an object, skipping: %r", raw)
            continue
        comp_id = raw.get("composition")
        spec = catalog.get(comp_id)
        if spec is None:
            log.warning("LLM graphics beat references unknown composition %r, skipping", comp_id)
            continue
        variables = raw.get("variables")
        if not isinstance(variables, dict):
            log.warning("LLM graphics beat for %r has no variables object, skipping", comp_id)
            continue
        str_variables = {str(k): str(v) for k, v in variables.items()}
        missing = spec.variable_ids - set(str_variables)
        if missing:
            log.warning("LLM graphics beat for %r missing variable(s) %s, skipping", comp_id, sorted(missing))
            continue
        anchor = str(raw.get("anchor_word", ""))
        anchor_word_obj = _snap_anchor(anchor, words)
        if anchor_word_obj is None:
            log.warning("LLM graphics beat anchor_word %r did not match any transcript word, skipping", anchor)
            continue
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        beat = GraphicsBeat(
            composition=comp_id,
            variables=str_variables,
            anchor_word=anchor,
            start=anchor_word_obj.start,
            duration=spec.duration,
            reason=str(raw.get("reason", "")),
        )
        candidates.append((confidence, beat))

    if not candidates:
        return GraphicsPlan(clip_id=clip_id, method="skipped", skipped_reason="LLM proposed no usable beats")

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected = [beat for _, beat in candidates[:max_beats]]
    selected.sort(key=lambda b: b.start)
    return GraphicsPlan(clip_id=clip_id, beats=selected, method="llm")


def _beat_cache_key(beat: GraphicsBeat, fps: int) -> str:
    """Deterministic: identical (composition, variables, duration, fps)
    always hashes the same, so a re-run skips the render entirely."""
    payload = json.dumps(
        {"composition": beat.composition, "variables": beat.variables, "duration": beat.duration, "fps": fps},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_beat(
    beat: GraphicsBeat,
    spec: CompositionSpec,
    graphics_dir: Path,
    cache_dir: Path,
    npx_bin: str = "npx",
    fps: int = DEFAULT_FPS,
    timeout: float = DEFAULT_RENDER_TIMEOUT_S,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{_beat_cache_key(beat, fps)}.mov"
    if out_path.exists():
        return out_path  # deterministic cache hit

    cmd = [
        npx_bin, "hyperframes", "render",
        "-c", spec.file,
        "-o", str(out_path.resolve()),
        "--format", "mov",
        "--fps", str(fps),
        "--strict", "--strict-variables",
        "--variables", json.dumps(beat.variables),
    ]
    run_or_raise(cmd, f"graphics render ({beat.composition})", cwd=graphics_dir, timeout=timeout)
    return out_path


def _composite(clip_path: Path, rendered_beats: list[tuple[GraphicsBeat, Path]], out_path: Path) -> Path:
    """One ffmpeg pass, N chained overlays -- each gated to its own
    [start, start+duration) window via enable='between(t,S,S+D)', never
    touching the base video outside that window. -map 0:a? (optional map)
    so an audio-less clip doesn't fail the composite."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(clip_path)]
    for _, path in rendered_beats:
        cmd += ["-i", str(path)]

    filter_parts = []
    prev_label = "0:v"
    for i, (beat, _) in enumerate(rendered_beats, start=1):
        out_label = f"ov{i}"
        window_end = beat.start + beat.duration
        filter_parts.append(
            f"[{prev_label}][{i}:v]overlay=0:0:enable='between(t,{beat.start:.3f},{window_end:.3f})'[{out_label}]"
        )
        prev_label = out_label

    cmd += [
        "-filter_complex", ";".join(filter_parts),
        "-map", f"[{prev_label}]", "-map", "0:a?",
        "-c:a", "copy",
        str(out_path),
    ]
    run_or_raise(cmd, "graphics composite")
    return out_path


def plan_and_render_graphics(
    clip_id: str,
    clip_path: Path,
    clip_words: list[Word],
    out_path: Path,
    graphics_dir: Path | None = None,
    npx_bin: str = "npx",
    max_beats: int = GRAPHICS_MAX_BEATS_DEFAULT,
    fps: int = DEFAULT_FPS,
    render_timeout_s: float = DEFAULT_RENDER_TIMEOUT_S,
) -> GraphicsPlan:
    """Always writes out_path -- a straight copy of clip_path on any
    skip/failure short of the final composite. See module docstring for the
    full degrade ladder."""
    graphics_dir = graphics_dir or DEFAULT_GRAPHICS_DIR
    out_path.parent.mkdir(parents=True, exist_ok=True)

    available, reason = graphics_available(graphics_dir, npx_bin=npx_bin)
    if not available:
        shutil.copy2(clip_path, out_path)
        return GraphicsPlan(clip_id=clip_id, method="skipped", skipped_reason=reason)

    client = get_llm_client()
    if client is None:
        shutil.copy2(clip_path, out_path)
        return GraphicsPlan(clip_id=clip_id, method="skipped", skipped_reason="no LLM key configured")

    catalog = load_catalog(graphics_dir)
    try:
        plan = _plan_beats_llm(client, clip_id, clip_words, catalog, max_beats)
    except Exception as exc:  # noqa: BLE001 - any LLM/parsing failure degrades to skipped, never crashes
        log.warning("LLM graphics planning failed (%s); skipping graphics for clip %s", exc, clip_id)
        shutil.copy2(clip_path, out_path)
        return GraphicsPlan(clip_id=clip_id, method="skipped", skipped_reason=f"LLM planning failed: {exc}")

    if plan.method == "skipped" or not plan.beats:
        shutil.copy2(clip_path, out_path)
        return plan

    cache_dir = graphics_dir / "renders"
    rendered: list[tuple[GraphicsBeat, Path]] = []
    for beat in plan.beats:
        spec = catalog.get(beat.composition)
        if spec is None:  # defensive -- _plan_beats_llm already validated this
            continue
        try:
            path = _render_beat(beat, spec, graphics_dir, cache_dir, npx_bin=npx_bin, fps=fps, timeout=render_timeout_s)
        except Exception as exc:  # noqa: BLE001 - one beat failing must not sink the batch
            log.warning("graphics beat render failed (%s, composition=%s); dropping this beat", exc, beat.composition)
            continue
        rendered.append((beat, path))

    if not rendered:
        shutil.copy2(clip_path, out_path)
        return plan.model_copy(update={"beats": []})

    # Only the final composite failing is a real error (propagates as-is --
    # the caller wraps it into a PipelineError, same as every other stage).
    _composite(clip_path, rendered, out_path)
    return plan.model_copy(update={"beats": [b for b, _ in rendered]})
