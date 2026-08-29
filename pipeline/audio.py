"""Audio enhancement -- the `enhance` stage, runs once per source before
transcribe. Zero-key, ffmpeg-only (Descript's "Studio Sound" is the one
real Descript differentiator Herk's own demonstrated pipeline doesn't
have -- see the project's plan notes). Denoise -> de-ess -> speech-level
-> limit -> two-pass loudnorm to a target LUFS. Video is always -c:v copy;
only the audio is re-encoded.

Runs once on the whole source, not per clip: a per-clip loudnorm would
give every clip a different perceived volume, and normalized/denoised
audio measurably improves whisper transcription accuracy for everything
downstream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.procutil import SubprocessFailedError, run_or_raise

log = logging.getLogger(__name__)

DEFAULT_TARGET_LUFS = -16.0


def _build_filter_chain(rnnoise_model_path: Path | None) -> str:
    """afftdn is the zero-key default denoiser (no model file needed).
    arnndn (deeper denoise) is opt-in only -- it requires an external
    .rnnn model file ffmpeg does not ship, verified via
    `ffmpeg -h filter=arnndn` (its `model` argument is required)."""
    parts = ["highpass=f=80"]
    if rnnoise_model_path is not None:
        parts.append(f"arnndn=model={rnnoise_model_path}")
    else:
        parts.append("afftdn=nf=-25")
    parts.append("deesser")
    parts.append("speechnorm=e=6.25:r=0.0001:l=1")
    parts.append("alimiter=limit=0.95")
    return ",".join(parts)


def _extract_json_stats(stderr: str) -> dict | None:
    """ffmpeg's loudnorm filter prints its JSON stats block to stderr,
    interleaved with other progress text -- pull out the last balanced
    {...} it wrote. Returns None (never raises) on anything unparseable,
    the same tolerance posture analyst.py's CSV handling uses."""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None


def _measure_loudness(source: Path, chain: str) -> dict | None:
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-af", f"{chain},loudnorm=print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = run_or_raise(cmd, "audio loudness measurement")
    except SubprocessFailedError as exc:
        log.warning("loudness measurement pass failed (%s); falling back to single-pass loudnorm", exc)
        return None
    return _extract_json_stats(proc.stderr or "")


def enhance(
    source: Path,
    out_path: Path,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    rnnoise_model_path: Path | None = None,
) -> float | None:
    """Runs the enhancement chain, writing out_path. Returns the achieved
    integrated loudness in LUFS if ffmpeg reported it, else None (enhance
    still succeeded -- the measurement is informational, stored on
    MediaAsset.loudness_lufs, not required for the pipeline to proceed)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chain = _build_filter_chain(rnnoise_model_path)

    measured = _measure_loudness(source, chain)
    loudnorm = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json"
    if measured is not None:
        try:
            loudnorm = (
                f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
                f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
                f"linear=true:print_format=json"
            )
        except KeyError:
            log.warning("loudness measurement JSON missing expected keys; falling back to single-pass loudnorm")

    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-af", f"{chain},{loudnorm}",
        "-c:v", "copy",
        str(out_path),
    ]
    proc = run_or_raise(cmd, "audio enhance")

    stats = _extract_json_stats(proc.stderr or "")
    if stats is None:
        return None
    try:
        return float(stats.get("output_i", stats.get("input_i")))
    except (TypeError, ValueError):
        return None
