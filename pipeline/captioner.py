"""Raw clip mp4 + word timestamps -> finished mp4 with burned-in captions.

Three styles, all zero-dependency (libass ships with this box's ffmpeg
build -- confirmed via `ffmpeg -filters`/`-enable-libass`, no new
dependency needed for any of them):

- "plain" (default): one SRT cue per few words, burned in via
  `subtitles=...:force_style=...`. The original, narrow-interface renderer
  (clip in, words in, finished mp4 out) -- deliberately kept as a drop-in
  swap target for a future Remotion-based renderer, see the README's
  "Swapping in Remotion later" section.
- "karaoke": a full ASS file with per-word `{\\k}` highlight tags, one
  Dialogue per cue -- the classic word-by-word karaoke fill.
- "pop": a full ASS file with one Dialogue PER WORD, each with a
  `\\pos`+`\\t(...)fscx/fscy` scale-pop entrance -- the actual
  TikTok-caption look.

karaoke/pop write an .ass sidecar and burn in via a bare `subtitles=...`
(no force_style -- style lives in the file's own [V4+ Styles] section, and
a force_style override would fight with the per-word \\k/\\t tags).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.procutil import run_or_raise
from pipeline.schemas import Word

# ASS force_style string used by the "plain" (SRT) style: white text, black
# outline/box, bottom-centered.
DEFAULT_STYLE = (
    "FontSize=22,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,"
    "BorderStyle=3,Outline=2,Shadow=0,Alignment=2,MarginV=60"
)

MAX_WORDS_PER_CUE = 6

# Canvas ASS positions/sizes are expressed against -- libass scales this to
# the actual video frame via PlayResX/PlayResY, so karaoke/pop don't need to
# know a clip's real resolution.
ASS_PLAY_RES = (384, 288)
POP_ANCHOR = (ASS_PLAY_RES[0] // 2, ASS_PLAY_RES[1] - 40)  # bottom-center


def _format_srt_time(t: float) -> str:
    """Integer-millisecond arithmetic throughout -- the previous float-based
    version (`int(round((t - int(t)) * 1000))`) could round up to 1000 with
    no carry into the seconds field (e.g. "00:00:01,1000"), reachable via
    ordinary 2-decimal faster-whisper timestamps. divmod on a single
    rounded integer has no such edge case."""
    total_ms = int(round(max(0.0, t) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _format_ass_time(t: float) -> str:
    """Same integer-arithmetic discipline as _format_srt_time, at ASS's
    centisecond resolution (H:MM:SS.CC, single-digit hour field)."""
    total_cs = int(round(max(0.0, t) * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    seconds, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def _group_into_cues(words: list[Word], max_words_per_cue: int) -> list[list[Word]]:
    """Shared by words_to_srt and words_to_ass's karaoke style: groups words
    into cues by count or sentence-ending punctuation."""
    cues: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        if len(current) >= max_words_per_cue or w.text.strip().endswith((".", "!", "?")):
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def words_to_srt(words: list[Word], offset: float = 0.0, max_words_per_cue: int = MAX_WORDS_PER_CUE) -> str:
    """Groups words into short caption cues (by count or sentence-ending
    punctuation) and renders standard SRT. `offset` shifts absolute
    transcript timestamps back to clip-relative time (usually clip.start)."""
    if not words:
        return ""

    cues = _group_into_cues(words, max_words_per_cue)

    lines = []
    for idx, cue_words in enumerate(cues, start=1):
        start = _format_srt_time(cue_words[0].start - offset)
        end_val = max(cue_words[-1].end - offset, cue_words[0].start - offset + 0.2)
        end = _format_srt_time(end_val)
        text = " ".join(w.text for w in cue_words)
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _escape_ass_text(text: str) -> str:
    """ASS override tags live inside {}; strip characters that would corrupt
    the tag stream. Safe/lossy-only for ordinary caption words -- mirrors
    the "clip ids are alnum+hyphen, so no escaping needed" posture already
    used elsewhere in this module, just applied to caption text instead."""
    return text.replace("\\", "").replace("{", "").replace("}", "")


def _k_durations(words: list[Word], line_start: float) -> list[int]:
    """Per-word `\\k` duration in integer centiseconds, computed by
    differencing CUMULATIVE ROUNDED offsets (each word's end-time-since-
    line-start, rounded once) rather than rounding each word's own duration
    independently. Independent rounding drifts over a long line -- the same
    bug class _format_srt_time's docstring above already describes and
    fixes for SRT; \\k timing is cumulative from the Dialogue's start, so it
    is exactly as vulnerable. A trailing silence before word i is folded
    into word i's own highlight duration (there's no separate "gap" token),
    which also keeps the sum of durations exactly equal to the line's total
    rounded length -- required for the highlight to reach the last word
    exactly when the line ends."""
    cum_cs = [0]
    for w in words:
        cum_cs.append(round((w.end - line_start) * 100))
    return [max(1, cum_cs[i + 1] - cum_cs[i]) for i in range(len(words))]


def _ass_header(style_name: str) -> str:
    primary = "&H00FFFFFF&"  # unhighlighted text: white
    highlight = "&H0000D7FF&"  # BGR order -- gold, the "already sung" karaoke color
    outline = "&H00000000&"
    back = "&H00000000&"
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {ASS_PLAY_RES[0]}\n"
        f"PlayResY: {ASS_PLAY_RES[1]}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {style_name},Arial,22,{primary},{highlight},{outline},{back},"
        "0,0,0,0,100,100,0,0,3,2,0,2,10,10,60,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _karaoke_dialogue_lines(cues: list[list[Word]], offset: float, style_name: str) -> list[str]:
    lines = []
    for cue_words in cues:
        line_start = cue_words[0].start - offset
        line_end = max(cue_words[-1].end - offset, line_start + 0.2)
        durations_cs = _k_durations(cue_words, cue_words[0].start)
        text = "".join(
            f"{{\\k{dur_cs}}}{_escape_ass_text(w.text)} " for w, dur_cs in zip(cue_words, durations_cs)
        ).rstrip()
        lines.append(f"Dialogue: 0,{_format_ass_time(line_start)},{_format_ass_time(line_end)},{style_name},,0,0,0,,{text}")
    return lines


def _pop_dialogue_lines(words: list[Word], offset: float, style_name: str) -> list[str]:
    lines = []
    x, y = POP_ANCHOR
    for w in words:
        start = w.start - offset
        end = max(w.end - offset, start + 0.12)
        dur_ms = max(60, int(round((end - start) * 1000)))
        up_ms = min(120, dur_ms // 2)
        settle_ms = min(up_ms + 120, dur_ms)
        override = (
            f"\\an5\\pos({x},{y})"
            f"\\fscx60\\fscy60\\t(0,{up_ms},\\fscx110\\fscy110)\\t({up_ms},{settle_ms},\\fscx100\\fscy100)"
        )
        text = _escape_ass_text(w.text)
        lines.append(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},{style_name},,0,0,0,,{{{override}}}{text}")
    return lines


def words_to_ass(words: list[Word], offset: float = 0.0, style: str = "karaoke", max_words_per_cue: int = MAX_WORDS_PER_CUE) -> str:
    """Full [V4+ Styles] ASS file. style="karaoke": one Dialogue per cue with
    per-word {\\k} highlight tags (see _k_durations for the drift-free
    timing). style="pop": one Dialogue per word with a \\pos+\\t(...)fscx/
    fscy scale-pop entrance. Empty `words` returns ""."""
    if not words:
        return ""
    if style not in ("karaoke", "pop"):
        raise ValueError(f"words_to_ass: unknown style {style!r} (expected 'karaoke' or 'pop')")

    style_name = "Default"
    header = _ass_header(style_name)
    if style == "karaoke":
        cues = _group_into_cues(words, max_words_per_cue)
        lines = _karaoke_dialogue_lines(cues, offset, style_name)
    else:
        lines = _pop_dialogue_lines(words, offset, style_name)
    return header + "\n".join(lines) + "\n"


def render_captioned_clip(
    clip_path: Path,
    words: list[Word],
    out_path: Path,
    offset: float = 0.0,
    style: str = "plain",
    force_style_override: str | None = None,
) -> Path:
    """style: "plain" (SRT + force_style, the original renderer),
    "karaoke", or "pop" (both ASS -- see module docstring).
    force_style_override only applies to "plain"."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if style not in ("plain", "karaoke", "pop"):
        raise ValueError(f"render_captioned_clip: unknown style {style!r} (expected 'plain', 'karaoke', or 'pop')")

    sub_content = words_to_srt(words, offset=offset) if style == "plain" else words_to_ass(words, offset=offset, style=style)

    if not sub_content.strip():
        # No words for this clip (e.g. silent b-roll) -- pass the clip
        # through untouched rather than failing the pipeline.
        shutil.copy2(clip_path, out_path)
        return out_path

    clip_path = clip_path.resolve()
    out_path = out_path.resolve()
    sub_suffix = ".srt" if style == "plain" else ".ass"
    sub_path = clip_path.with_suffix(sub_suffix)
    sub_path.write_text(sub_content, encoding="utf-8")

    # Run with cwd set to the subtitle's own directory and reference it by
    # bare filename -- sidesteps ffmpeg's subtitles-filter path escaping
    # entirely (a DATA_DIR containing a colon or apostrophe used to break
    # the filtergraph parse no matter how carefully the path was escaped).
    # Safe because clip ids are run-scoped alnum+hyphen (see
    # clip_selector.py), so the filename itself never contains special
    # characters; -i/output stay absolute so they're unaffected by the cwd
    # change.
    if style == "plain":
        force_style = force_style_override or DEFAULT_STYLE
        vf = f"subtitles={sub_path.name}:force_style='{force_style}'"
    else:
        # karaoke/pop style lives IN the ASS file's own [V4+ Styles]
        # section -- a force_style override here would fight with the
        # per-word \k/\t override tags baked into each Dialogue line.
        vf = f"subtitles={sub_path.name}"

    cmd = ["ffmpeg", "-y", "-i", str(clip_path), "-vf", vf, "-c:a", "copy", str(out_path)]
    run_or_raise(cmd, "caption burn-in", cwd=sub_path.parent)
    return out_path
