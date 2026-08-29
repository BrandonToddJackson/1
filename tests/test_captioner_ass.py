"""Pure-Python tests for captioner.py's ASS karaoke/pop rendering: the
drift-free \\k timing (_k_durations), _format_ass_time's brute-force
regression (same bug class _format_srt_time already had), and words_to_ass's
structural output. Real-ffmpeg rendering-through-libass tests live in
test_media.py alongside the rest of this module's real-ffmpeg coverage."""

import re

from pipeline import captioner
from pipeline.schemas import Word


# ---------------------------------------------------------------------------
# _format_ass_time
# ---------------------------------------------------------------------------

def test_format_ass_time_basic():
    assert captioner._format_ass_time(0.0) == "0:00:00.00"
    assert captioner._format_ass_time(61.5) == "0:01:01.50"
    assert captioner._format_ass_time(3661.25) == "1:01:01.25"


def test_format_ass_time_never_emits_100_centis():
    # Regression for the same float-rounding-carry bug class
    # _format_srt_time's docstring documents: brute force every hundredth
    # of a second up to 100s and confirm the centisecond field never
    # overflows to "100" with no carry into seconds.
    for hundredths in range(0, 10000):
        t = hundredths / 100
        formatted = captioner._format_ass_time(t)
        cs_part = formatted.split(".")[1]
        assert len(cs_part) == 2, f"t={t} produced malformed time {formatted!r}"
        assert int(cs_part) < 100, f"t={t} produced overflowed centiseconds {formatted!r}"


# ---------------------------------------------------------------------------
# _k_durations -- drift-free cumulative-rounding \k timing
# ---------------------------------------------------------------------------

def test_k_durations_sums_to_total_rounded_duration():
    words = [
        Word(text="one", start=10.0, end=10.31),
        Word(text="two", start=10.4, end=10.67),
        Word(text="three", start=10.7, end=11.09),
    ]
    line_start = words[0].start
    durations = captioner._k_durations(words, line_start)
    expected_total = round((words[-1].end - line_start) * 100)
    assert sum(durations) == expected_total


def test_k_durations_all_positive():
    words = [Word(text="a", start=0.0, end=0.001), Word(text="b", start=0.001, end=0.002)]
    durations = captioner._k_durations(words, 0.0)
    assert all(d >= 1 for d in durations)


def test_k_durations_never_drifts_over_many_random_words():
    import random

    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(1, 30)
        t = 0.0
        words = []
        for i in range(n):
            t += rng.uniform(0.05, 0.6)
            start = t
            t += rng.uniform(0.05, 0.4)
            end = t
            words.append(Word(text=f"w{i}", start=start, end=end))
        line_start = words[0].start
        durations = captioner._k_durations(words, line_start)
        assert sum(durations) == round((words[-1].end - line_start) * 100)
        assert all(isinstance(d, int) and d >= 1 for d in durations)


# ---------------------------------------------------------------------------
# words_to_ass -- structural output
# ---------------------------------------------------------------------------

def _sample_words():
    return [
        Word(text="hello", start=1.0, end=1.3),
        Word(text="world.", start=1.4, end=1.7),
        Word(text="next", start=2.0, end=2.3),
    ]


def test_words_to_ass_empty_returns_empty_string():
    assert captioner.words_to_ass([], style="karaoke") == ""
    assert captioner.words_to_ass([], style="pop") == ""


def test_words_to_ass_unknown_style_raises():
    import pytest

    with pytest.raises(ValueError):
        captioner.words_to_ass(_sample_words(), style="bogus")


def test_words_to_ass_karaoke_has_header_and_k_tags():
    ass = captioner.words_to_ass(_sample_words(), style="karaoke")
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Dialogue:" in ass
    assert re.search(r"\{\\k\d+\}hello", ass)
    assert "world" in ass


def test_words_to_ass_karaoke_breaks_cue_on_sentence_end():
    ass = captioner.words_to_ass(_sample_words(), style="karaoke", max_words_per_cue=10)
    # "world." ends a sentence -- "next" must start a NEW Dialogue line even
    # though max_words_per_cue (10) was never reached.
    assert ass.count("Dialogue:") == 2


def test_words_to_ass_pop_has_one_dialogue_per_word():
    ass = captioner.words_to_ass(_sample_words(), style="pop")
    assert ass.count("Dialogue:") == 3
    assert "\\pos(" in ass
    assert "\\fscx" in ass and "\\fscy" in ass
    assert "\\t(" in ass


def test_words_to_ass_offset_shifts_timestamps():
    words = [Word(text="hi", start=10.0, end=10.5)]
    ass = captioner.words_to_ass(words, offset=10.0, style="pop")
    assert "0:00:00.00,0:00:00.50" in ass


def test_words_to_ass_escapes_ass_special_characters():
    words = [Word(text="{brace}\\slash", start=0.0, end=0.3)]
    ass = captioner.words_to_ass(words, offset=0.0, style="pop")
    # Only the override-tag braces should survive -- the word's own braces
    # and backslash must be stripped so they can't corrupt the tag stream.
    dialogue_line = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")][0]
    text_field = dialogue_line.split(",", 9)[-1]
    # The override tag block is the leading {...}; whatever follows it is
    # the actual caption text and must be free of stray braces/backslashes.
    assert text_field.count("{") == 1 and text_field.count("}") == 1


# ---------------------------------------------------------------------------
# render_captioned_clip style validation (no ffmpeg needed for this part)
# ---------------------------------------------------------------------------

def test_render_captioned_clip_rejects_unknown_style(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        captioner.render_captioned_clip(
            tmp_path / "in.mp4", _sample_words(), tmp_path / "out.mp4", style="bogus",
        )
