"""Raw clip mp4 + word timestamps -> finished mp4 with burned-in captions.

This is the zero-dependency default renderer, deliberately kept to a narrow
interface (clip in, words in, finished mp4 out) so it's a drop-in swap for a
future Remotion-based renderer without touching any other stage -- see the
README's "Swapping in Remotion later" section.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pipeline.schemas import Word

# ASS force_style string: white text, black outline/box, bottom-centered.
DEFAULT_STYLE = (
    "FontSize=22,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,"
    "BorderStyle=3,Outline=2,Shadow=0,Alignment=2,MarginV=60"
)

MAX_WORDS_PER_CUE = 6


def _format_srt_time(t: float) -> str:
    t = max(0.0, t)
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def words_to_srt(words: list[Word], offset: float = 0.0, max_words_per_cue: int = MAX_WORDS_PER_CUE) -> str:
    """Groups words into short caption cues (by count or sentence-ending
    punctuation) and renders standard SRT. `offset` shifts absolute
    transcript timestamps back to clip-relative time (usually clip.start)."""
    if not words:
        return ""

    cues: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        if len(current) >= max_words_per_cue or w.text.strip().endswith((".", "!", "?")):
            cues.append(current)
            current = []
    if current:
        cues.append(current)

    lines = []
    for idx, cue_words in enumerate(cues, start=1):
        start = _format_srt_time(cue_words[0].start - offset)
        end_val = max(cue_words[-1].end - offset, cue_words[0].start - offset + 0.2)
        end = _format_srt_time(end_val)
        text = " ".join(w.text for w in cue_words)
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _escape_subtitles_path(path: Path) -> str:
    # ffmpeg's subtitles filter treats ':' as an option separator even inside
    # a quoted filename; escape it. Backslashes need escaping too.
    s = str(path).replace("\\", "\\\\").replace(":", "\\:")
    return s


def render_captioned_clip(
    clip_path: Path,
    words: list[Word],
    out_path: Path,
    offset: float = 0.0,
    style: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    srt_content = words_to_srt(words, offset=offset)

    if not srt_content.strip():
        # No words for this clip (e.g. silent b-roll) -- pass the clip
        # through untouched rather than failing the pipeline.
        shutil.copy2(clip_path, out_path)
        return out_path

    srt_path = clip_path.with_suffix(".srt")
    srt_path.write_text(srt_content, encoding="utf-8")

    force_style = style or DEFAULT_STYLE
    vf = f"subtitles='{_escape_subtitles_path(srt_path)}':force_style='{force_style}'"
    cmd = ["ffmpeg", "-y", "-i", str(clip_path), "-vf", vf, "-c:a", "copy", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path
