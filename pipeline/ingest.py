"""Source (YouTube/URL or local file path) -> MediaAsset.

Always deterministic, no API key required: yt-dlp handles URL downloads for
free, local files are just copied/referenced. ffmpeg's `ffprobe` is used to
read duration.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from pipeline.procutil import SubprocessFailedError, run_or_raise
from pipeline.schemas import MediaAsset
from pipeline.storage import run_dir

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

# Sidecar files yt-dlp can leave behind (a previous interrupted download, or
# extra flags like --write-subs/-k/--write-info-json) that must never be
# mistaken for the actual downloaded media when falling back to a glob.
_SIDECAR_SUFFIXES = (
    ".part", ".ytdl", ".info.json", ".vtt", ".srt",
    ".jpg", ".jpeg", ".png", ".webp", ".description",
)


def is_url(source: str) -> bool:
    return bool(_URL_RE.match(source))


def ingest(source: str, run_id: str) -> MediaAsset:
    """source: a local file path, or a URL yt-dlp can handle. Always writes
    the resulting media into runs/<run_id>/source.<ext>."""
    dest_dir = run_dir(run_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        local_path = _download_with_ytdlp(source, dest_dir)
    else:
        src_path = Path(source).expanduser().resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Local media file not found: {src_path}")
        local_path = dest_dir / f"source{src_path.suffix or '.mp4'}"
        shutil.copy2(src_path, local_path)

    info = _probe_media(local_path)
    return MediaAsset(
        run_id=run_id,
        source=source,
        local_path=str(local_path),
        duration=info.get("duration", 0.0),
        title=info.get("title"),
    )


def _download_with_ytdlp(url: str, dest_dir: Path) -> Path:
    out_template = str(dest_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "mp4/bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--print", "after_move:filepath",
        "-o", out_template,
        url,
    ]
    result = run_or_raise(cmd, "yt-dlp download")

    # Prefer yt-dlp's own report of the final output path over guessing --
    # this is the only reliable way to know which file is the real download
    # when sidecar files (partial fragments, subtitles, thumbnails) can
    # share the "source.*" prefix.
    printed_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if printed_lines:
        printed_path = Path(printed_lines[-1])
        if printed_path.exists():
            return printed_path

    # Fallback: --print gave nothing usable. Glob, but explicitly exclude
    # known sidecar suffixes rather than blindly taking the alphabetically
    # first match (which can pick a ".info.json" or ".part" file over the
    # real video).
    matches = sorted(
        p for p in dest_dir.glob("source.*") if not any(p.name.endswith(suffix) for suffix in _SIDECAR_SUFFIXES)
    )
    if not matches:
        raise RuntimeError(f"yt-dlp reported success but no output file was found in {dest_dir}")
    return matches[0]


def _probe_media(path: Path) -> dict:
    """Runs ffprobe. A missing ffprobe binary is an environment problem --
    MissingBinaryError propagates so ingest fails fast instead of silently
    reporting duration=0.0 "success" (the real problem previously only
    surfaced much later, at the `cut` stage). A file ffprobe simply can't
    parse is non-fatal and returns {}."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    try:
        proc = run_or_raise(cmd, "ffprobe")
    except SubprocessFailedError:
        return {}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    fmt = data.get("format", {})
    result: dict = {}
    if "duration" in fmt:
        result["duration"] = float(fmt["duration"])
    tags = fmt.get("tags", {})
    if "title" in tags:
        result["title"] = tags["title"]
    return result
