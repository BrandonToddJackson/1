"""Subprocess wrapper shared by ingest.py/cutter.py/captioner.py.

The bare `subprocess.run(cmd, check=True, capture_output=True)` pattern used
everywhere ffmpeg/ffprobe/yt-dlp are invoked loses the real stderr on
failure -- CalledProcessError's default str() is just
"Command '[...]' returned non-zero exit status 1.", with the actual reason
thrown away, and a missing binary raises a raw FileNotFoundError with no
guidance. run_or_raise() fixes both.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class MissingBinaryError(RuntimeError):
    """The command itself isn't installed / not on PATH."""


class SubprocessFailedError(RuntimeError):
    """The command ran but exited non-zero; includes its real stderr."""


def run_or_raise(cmd: list[str], step_label: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=cwd)
    except FileNotFoundError as exc:
        raise MissingBinaryError(
            f"{step_label}: required binary not found: {cmd[0]!r} "
            f"(is it installed and on PATH? e.g. apt-get install -y ffmpeg)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        tail = "\n".join(stderr.splitlines()[-20:]) if stderr else "(no stderr output)"
        raise SubprocessFailedError(f"{step_label} failed (exit {exc.returncode}): {tail}") from exc
