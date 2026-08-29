from pipeline import textutils


def test_truncate_returns_short_text_unchanged():
    assert textutils.truncate("hello", 100) == "hello"


def test_truncate_preserves_word_boundary():
    truncated = textutils.truncate("one two three four five", 10)
    assert truncated.endswith("…")
    assert not truncated[:-1].endswith(" ")


def test_truncate_zero_or_negative_returns_empty():
    assert textutils.truncate("anything", 0) == ""
    assert textutils.truncate("anything", -5) == ""


def test_truncate_exact_length_boundary():
    assert textutils.truncate("12345", 5) == "12345"
