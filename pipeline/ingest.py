"""Source (YouTube/URL or local file path) -> MediaAsset.

Always deterministic, no API key required: yt-dlp handles URL downloads for
free, local files are just copied/referenced. ffmpeg's `ffprobe` is used to
read duration.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from pipeline.schemas import MediaAsset
from pipeline.storage import run_dir

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


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
        "-o", out_template,
        url,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    matches = sorted(dest_dir.glob("source.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp reported success but no output file was found in {dest_dir}")
    return matches[0]


def _probe_media(path: Path) -> dict:
    """Runs ffprobe; returns {} (rather than raising) if ffprobe is
    unavailable or the file can't be probed, so ingest never hard-fails on
    metadata alone."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
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
