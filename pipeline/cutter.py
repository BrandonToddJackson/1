"""Source media + Clip -> raw cut mp4, via ffmpeg subprocess.

Seeks with `-i` before `-ss`/`-to` so cuts land on the exact word-timestamp
boundaries from the transcript rather than the nearest keyframe (a plain
stream-copy `-ss`-before-`-i` cut can drift by up to a couple of seconds).
That costs a re-encode, which is fine at clip length (<=~90s).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.procutil import run_or_raise
from pipeline.schemas import Clip


def cut_clip(source: Path, clip: Clip, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip.id}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-ss", str(clip.start),
        "-to", str(clip.end),
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
        str(out_path),
    ]
    run_or_raise(cmd, f"cut {clip.id}")
    return out_path


def cut_all(source: Path, clips: list[Clip], out_dir: Path) -> dict[str, Path]:
    return {clip.id: cut_clip(source, clip, out_dir) for clip in clips}
