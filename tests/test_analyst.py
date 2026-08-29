from pathlib import Path

import pytest

from pipeline import analyst
from pipeline.schemas import Clip, PerformanceRecord, Post

FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "sample_performance.csv"


@pytest.fixture
def records() -> list[PerformanceRecord]:
    return analyst.load_performance(FIXTURE_CSV)


@pytest.fixture
def clips() -> list[Clip]:
    return [
        Clip(id="clip-01", start=0, end=30, hook="the biggest mistake with money",
             topic="mistake money saving", score=0.8, selection_method="heuristic"),
        Clip(id="clip-02", start=30, end=70, hook="the secret to compound interest",
             topic="secret compound interest", score=0.9, selection_method="heuristic"),
        Clip(id="clip-03", start=70, end=90, hook="a totally unrelated story",
             topic="story vegetables grocery", score=0.4, selection_method="heuristic"),
    ]


@pytest.fixture
def posts() -> list[Post]:
    return [
        Post(platform="instagram", clip_id="clip-02", text="...", hashtags=["#compoundinterest", "#investing"]),
        Post(platform="linkedin", clip_id="clip-01", text="...", hashtags=["#money", "#saving"]),
    ]


def test_load_performance_parses_csv(records):
    assert len(records) == 6
    assert records[0].post_id == "p-linkedin-01"
    assert records[0].platform == "linkedin"
    assert records[0].views == 5000


def test_load_performance_computes_engagement_rate_when_missing(records):
    linkedin_01 = next(r for r in records if r.post_id == "p-linkedin-01")
    assert linkedin_01.engagement_rate is None
    assert linkedin_01.compute_engagement_rate() == pytest.approx(0.1)


def test_analyze_empty_records_returns_placeholder_learnings():
    learnings = analyst.analyze([])
    assert learnings.top_keywords == []
    assert "No performance data" in learnings.notes


def test_analyze_ranks_best_platforms(records):
    learnings = analyst.analyze(records)
    assert learnings.best_platforms[0] == "instagram"
    assert set(learnings.best_platforms) == {"linkedin", "x", "threads", "instagram"}


def test_analyze_picks_top_clips_by_engagement(records, clips):
    learnings = analyst.analyze(records, clips=clips)
    # clip-02 (instagram, highest engagement) and clip-01 (linkedin, 2nd) should
    # dominate the derived hook patterns; clip-03 (never a top performer) should not.
    assert any("compound interest" in h for h in learnings.top_hook_patterns)
    assert any("mistake" in h for h in learnings.top_hook_patterns)
    assert not any("unrelated" in h for h in learnings.top_hook_patterns)


def test_analyze_ideal_clip_length_range_from_top_clips(records, clips):
    learnings = analyst.analyze(records, clips=clips)
    assert learnings.ideal_clip_length_range is not None
    lo, hi = learnings.ideal_clip_length_range
    assert lo <= hi


def test_analyze_prefers_post_hashtags_over_mined_keywords(records, clips, posts):
    learnings = analyst.analyze(records, posts=posts, clips=clips)
    # Hashtags on the top posts should appear, and appear before purely
    # text-mined keywords.
    assert "compoundinterest" in learnings.top_keywords
    assert "money" in learnings.top_keywords


def test_save_and_load_learnings_roundtrip(tmp_path, records, clips):
    path = tmp_path / "learnings.json"
    learnings = analyst.analyze(records, clips=clips)
    analyst.save_learnings(learnings, path)

    loaded = analyst.load_learnings(path)
    assert loaded is not None
    assert loaded.best_platforms == learnings.best_platforms
    assert loaded.top_keywords == learnings.top_keywords


def test_load_learnings_returns_none_when_missing(tmp_path):
    assert analyst.load_learnings(tmp_path / "does-not-exist.json") is None


def _write_csv(tmp_path, content: str, name: str = "perf.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding=encoding)
    return path


def test_load_performance_handles_utf8_bom(tmp_path):
    csv_text = "post_id,platform,views,likes,comments,shares\np1,x,100,10,1,1\n"
    path = tmp_path / "perf.csv"
    path.write_bytes(("﻿" + csv_text).encode("utf-8"))

    records = analyst.load_performance(path)
    assert len(records) == 1
    assert records[0].post_id == "p1"


def test_load_performance_strips_whitespace_in_headers_and_values(tmp_path):
    csv_text = "post_id, platform , views\np1, x , 100\n"
    path = _write_csv(tmp_path, csv_text)

    records = analyst.load_performance(path)
    assert records[0].platform == "x"
    assert records[0].views == 100


def test_load_performance_parses_thousands_separator(tmp_path):
    csv_text = "post_id,platform,views\np1,x,\"1,234\"\n"
    path = _write_csv(tmp_path, csv_text)

    records = analyst.load_performance(path)
    assert records[0].views == 1234


def test_load_performance_parses_float_style_int(tmp_path):
    csv_text = "post_id,platform,views\np1,x,500.0\n"
    path = _write_csv(tmp_path, csv_text)

    records = analyst.load_performance(path)
    assert records[0].views == 500


def test_load_performance_parses_percent_engagement_rate(tmp_path):
    csv_text = "post_id,platform,engagement_rate\np1,x,12.5%\n"
    path = _write_csv(tmp_path, csv_text)

    records = analyst.load_performance(path)
    assert records[0].engagement_rate == pytest.approx(0.125)


def test_load_performance_missing_required_column_raises_clear_error(tmp_path):
    csv_text = "platform,views\nx,100\n"
    path = _write_csv(tmp_path, csv_text)

    with pytest.raises(ValueError, match="post_id"):
        analyst.load_performance(path)


def test_load_performance_bad_number_names_file_line_column(tmp_path):
    csv_text = "post_id,platform,views\np1,x,not-a-number\n"
    path = _write_csv(tmp_path, csv_text)

    with pytest.raises(ValueError) as exc_info:
        analyst.load_performance(path)
    message = str(exc_info.value)
    assert "perf.csv" in message
    assert "row 2" in message
    assert "views" in message


def test_load_performance_absent_optional_columns_default_to_zero(tmp_path):
    csv_text = "post_id,platform\np1,x\n"
    path = _write_csv(tmp_path, csv_text)

    records = analyst.load_performance(path)
    assert records[0].views == 0
    assert records[0].likes == 0
    assert records[0].engagement_rate is None


def test_load_performance_missing_post_id_value_raises(tmp_path):
    csv_text = "post_id,platform,views\n,x,100\n"
    path = _write_csv(tmp_path, csv_text)

    with pytest.raises(ValueError, match="post_id"):
        analyst.load_performance(path)


def test_analyze_notes_clip_id_mismatch(records):
    # None of these clip_ids exist in the provided clips list -- simulates
    # running `analyze --run-id` against the wrong run's clips.json.
    wrong_run_clips = [
        Clip(id="other-run-clip-01", start=0, end=30, hook="x", topic="x", score=0.5),
    ]
    learnings = analyst.analyze(records, clips=wrong_run_clips)
    assert "not found in the provided clips" in learnings.notes


def test_analyze_ignores_zero_duration_clips_for_ideal_range(records):
    zero_duration_clips = [
        Clip(id="clip-01", start=10.0, end=10.0, hook="x", topic="x", score=0.5),
        Clip(id="clip-02", start=0.0, end=0.0, hook="x", topic="x", score=0.5),
    ]
    learnings = analyst.analyze(records, clips=zero_duration_clips)
    assert learnings.ideal_clip_length_range is None
