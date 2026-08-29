"""Central settings: env loading, paths, run-id generation.

Every value here is optional and has a sane default -- the pipeline must run
with a completely empty .env file.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env once, at import time, without overriding real env vars that are
# already set (e.g. in CI or a shell export).
load_dotenv(override=False)


class Settings(BaseModel):
    data_dir: Path = Path("runs")
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ayrshare_api_key: str | None = None


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
    )


def new_run_id() -> str:
    """Timestamp-prefixed id so runs sort chronologically on disk."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"
