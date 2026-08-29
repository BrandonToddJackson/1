"""Extracts PNG frames from a clip's final rendered output for visual
review -- the `preview` command (a non-stage command, sibling of
`analyze`/`qc`, not part of STAGE_ORDER). This is the deterministic half of
the iterate-and-preview loop: `preview` -> a human/Claude Code looks at the
PNGs -> hand-edit graphics_plans.json -> `graphics --only <clip-id>`
re-renders -> `preview` again. See README.md's "the subjective loop"
section for why the actual subjective judgment deliberately stays outside
an automated CLI loop.

Zero-key, always works: just ffmpeg against whatever `caption`/`graphics`
already produced.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.procutil import run_or_raise
from pipeline.schemas import GraphicsPlan

DEFAULT_EVENLY_SPACED_FRAMES = 5


def default_preview_timestamps(
    duration: float,
    graphics_plan: GraphicsPlan | None = None,
    n_evenly_spaced: int = DEFAULT_EVENLY_SPACED_FRAMES,
) -> list[float]:
    """n evenly-spaced frames across the clip, plus the midpoint of every
    graphics beat -- so every card is guaranteed visible in the contact
    sheet even if it lands between two evenly-spaced frames. Deduped and
    sorted."""
    if duration <= 0:
        return []

    if n_evenly_spaced <= 1:
        spaced = [duration / 2]
    else:
        step = duration / (n_evenly_spaced - 1)
        spaced = [min(i * step, max(duration - 0.01, 0.0)) for i in range(n_evenly_spaced)]

    beat_mids = []
    if graphics_plan:
        for beat in graphics_plan.beats:
            mid = beat.start + beat.duration / 2
            if 0 <= mid <= duration:
                beat_mids.append(mid)

    return sorted({round(t, 2) for t in spaced + beat_mids})


def extract_frame(clip_path: Path, timestamp: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(clip_path), "-frames:v", "1", str(out_path)]
    run_or_raise(cmd, f"preview frame at {timestamp:.2f}s")
    return out_path


def extract_frames(clip_path: Path, timestamps: list[float], out_dir: Path) -> list[Path]:
    return [extract_frame(clip_path, t, out_dir / f"frame_{t:06.2f}.png") for t in timestamps]
