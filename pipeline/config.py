"""Central settings: env loading, paths, run-id generation.

Every value here is optional and has a sane default -- the pipeline must run
with a completely empty .env file.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env once, at import time, without overriding real env vars that are
# already set (e.g. in CI or a shell export).
load_dotenv(override=False)

log = logging.getLogger(__name__)


class Settings(BaseModel):
    data_dir: Path = Path("runs")
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ayrshare_api_key: str | None = None
    blotato_api_key: str | None = None
    # Blotato's API requires a per-platform accountId on every post; a bare
    # API key alone isn't enough to publish. {"x": "acct_123", "linkedin": "acct_456", ...}
    blotato_account_ids: dict[str, str] = Field(default_factory=dict)
    # Audio enhancement (the `enhance` stage) -- zero-key by default.
    audio_target_lufs: float = -16.0
    # Opt-in deeper denoise: arnndn needs an external .rnnn model file ffmpeg
    # doesn't ship. None (default) means afftdn (zero-key, no model needed).
    rnnoise_model_path: Path | None = None
    # Diarization (see pipeline/transcribe.py::get_transcribe_backend).
    elevenlabs_api_key: str | None = None
    huggingface_token: str | None = None
    transcribe_backend: str = "auto"
    diarization_backend: str = "auto"
    # Motion graphics (the `graphics` stage) -- zero-key-skipped by default,
    # see pipeline/graphics.py::graphics_available.
    graphics_project_dir: str = "graphics"
    graphics_max_beats: int = 4
    npx_bin: str = "npx"
    render_timeout_s: float = 600.0


def _parse_float(raw: str | None, default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("expected a float, got %r; using default %s", raw, default)
        return default


def _parse_int(raw: str | None, default: int) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("expected an int, got %r; using default %s", raw, default)
        return default


def _parse_blotato_account_ids(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("BLOTATO_ACCOUNT_IDS is not valid JSON, ignoring: %r", raw)
        return {}
    if not isinstance(parsed, dict):
        log.warning("BLOTATO_ACCOUNT_IDS must be a JSON object, ignoring: %r", raw)
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def get_settings() -> Settings:
    """Read settings fresh from the environment every call.

    Reading lazily (rather than caching a module-level singleton) keeps this
    testable: tests can monkeypatch os.environ and immediately see the effect
    without needing to reset a cache.
    """
    return Settings(
        data_dir=Path(os.environ.get("DATA_DIR", "runs")),
        whisper_model=os.environ.get("WHISPER_MODEL", "base"),
        whisper_device=os.environ.get("WHISPER_DEVICE", "cpu"),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        ayrshare_api_key=os.environ.get("AYRSHARE_API_KEY") or None,
        blotato_api_key=os.environ.get("BLOTATO_API_KEY") or None,
        blotato_account_ids=_parse_blotato_account_ids(os.environ.get("BLOTATO_ACCOUNT_IDS")),
        audio_target_lufs=_parse_float(os.environ.get("AUDIO_TARGET_LUFS"), -16.0),
        rnnoise_model_path=(Path(p) if (p := os.environ.get("RNNOISE_MODEL_PATH")) else None),
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY") or None,
        huggingface_token=os.environ.get("HUGGINGFACE_TOKEN") or None,
        transcribe_backend=os.environ.get("TRANSCRIBE_BACKEND", "auto"),
        diarization_backend=os.environ.get("DIARIZATION_BACKEND", "auto"),
        graphics_project_dir=os.environ.get("GRAPHICS_PROJECT_DIR", "graphics"),
        graphics_max_beats=_parse_int(os.environ.get("GRAPHICS_MAX_BEATS"), 4),
        npx_bin=os.environ.get("NPX_BIN", "npx"),
        render_timeout_s=_parse_float(os.environ.get("RENDER_TIMEOUT_S"), 600.0),
    )


def new_run_id() -> str:
    """Timestamp-prefixed id so runs sort chronologically on disk."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"
