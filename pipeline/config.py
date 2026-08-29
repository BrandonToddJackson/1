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
    )


def new_run_id() -> str:
    """Timestamp-prefixed id so runs sort chronologically on disk."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"
