"""Source media + one-or-more (start, end) ranges -> a single cut mp4, via
ffmpeg's filter_complex.

cut_ranges() is the real implementation: `trim`/`atrim` + `concat`, driven
by a `-filter_complex_script` sidecar file (sidesteps ARG_MAX/shell-quoting
the same way captioner.py's .srt sidecar does -- also the only way to keep
the graph readable once it spans dozens of ranges). `trim`/`atrim` operate
on DECODED frames, so joins land on the exact requested timestamp --
strictly stronger than a plain `-ss`-after-`-i` seek, not a violation of
this module's frame-accuracy contract. A hard cut on video plus a short
`afade` in/out on audio at every internal join (never `acrossfade`, which
would shorten total duration and break the clean<->source timeline math's
invertibility in pipeline/timeline.py) keeps joins from sounding like a
click without changing the output's total duration.

cut_clip() is a thin single-range delegator kept for its exact original
signature (existing callers/tests depend on it). A single range has no
internal joins, so it emits no fades -- behaviorally unchanged from before
this module grew multi-range support.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from pipeline import timeline
from pipeline.procutil import SubprocessFailedError, run_or_raise
from pipeline.schemas import Clip

log = logging.getLogger(__name__)

DEFAULT_FADE_MS = 30.0
# Caps the filtergraph's size -- timeline.merge_small_gaps folds the
# closest-together ranges together (logged) until the count fits, rather
# than silently truncating and dropping content.
MAX_KEEP_RANGES = 200


def _has_audio_stream(source: Path) -> bool:
    """A file ffprobe can't read is treated as audio-less (SubprocessFailedError
    -> False), matching ingest._probe_media's tolerance posture. A missing
    ffprobe binary is an environment problem and propagates (MissingBinaryError)."""
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(source),
    ]
    try:
        proc = run_or_raise(cmd, "probe audio stream")
    except SubprocessFailedError:
        return False
    return bool(proc.stdout.strip())


def _build_filter_script(ranges: list[tuple[float, float]], has_audio: bool, fade_s: float) -> str:
    n = len(ranges)
    parts: list[str] = []
    for i, (start, end) in enumerate(ranges):
        parts.append(f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            duration = end - start
            chain = [f"[0:a]atrim=start={start:.6f}:end={end:.6f}", "asetpts=PTS-STARTPTS"]
            if i > 0:  # fades only at INTERNAL joins -- a single range emits none
                chain.append(f"afade=t=in:st=0:d={fade_s:.4f}")
            if i < n - 1:
                fade_start = max(0.0, duration - fade_s)
                chain.append(f"afade=t=out:st={fade_start:.4f}:d={fade_s:.4f}")
            parts.append(",".join(chain) + f"[a{i}]")

    if has_audio:
        joins = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{joins}concat=n={n}:v=1:a=1[outv][outa]")
    else:
        joins = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{joins}concat=n={n}:v=1:a=0[outv]")
    return ";\n".join(parts) + "\n"


def cut_ranges(
    source: Path,
    ranges: list[tuple[float, float]],
    out_path: Path,
    fade_ms: float = DEFAULT_FADE_MS,
) -> Path:
    """Extracts and concatenates `ranges` (source-timeline seconds) from
    `source` into a single file at `out_path`. Ranges need not be adjacent
    or in order on disk -- they're joined in the order given."""
    clean_ranges = [(s, e) for s, e in ranges if e - s > 1e-6]
    if not clean_ranges:
        raise ValueError("cut_ranges: at least one non-empty range is required")

    if len(clean_ranges) > MAX_KEEP_RANGES:
        log.warning(
            "cut_ranges: %d ranges exceeds MAX_KEEP_RANGES=%d, merging closest-together pairs",
            len(clean_ranges), MAX_KEEP_RANGES,
        )
    merged_ranges = timeline.merge_small_gaps(clean_ranges, MAX_KEEP_RANGES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_audio = _has_audio_stream(source)
    script = _build_filter_script(merged_ranges, has_audio, fade_ms / 1000.0)

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat.txt", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = Path(f.name)

    try:
        cmd = ["ffmpeg", "-y", "-i", str(source), "-filter_complex_script", str(script_path), "-map", "[outv]"]
        if has_audio:
            cmd += ["-map", "[outa]"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast"]
        if has_audio:
            cmd += ["-c:a", "aac"]
        cmd.append(str(out_path))
        run_or_raise(cmd, "cut_ranges")
    finally:
        script_path.unlink(missing_ok=True)

    return out_path


def cut_clip(source: Path, clip: Clip, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip.id}.mp4"
    return cut_ranges(source, [(clip.start, clip.end)], out_path)


def cut_all(source: Path, clips: list[Clip], out_dir: Path) -> dict[str, Path]:
    return {clip.id: cut_clip(source, clip, out_dir) for clip in clips}
