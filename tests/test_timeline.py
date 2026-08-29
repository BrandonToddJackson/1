import pytest

from pipeline import timeline
from pipeline.schemas import EditDecision, EditPlan, Transcript, TranscriptSegment, Word


def _identity_plan(duration: float) -> EditPlan:
    return EditPlan(
        run_id="r1",
        source_duration=duration,
        decisions=[EditDecision(start=0.0, end=duration, action="keep")],
        method="identity",
    )


def _gap_plan() -> EditPlan:
    # keep [0,10), remove [10,15), keep [15,25) -- 20s clean, 5s removed
    return EditPlan(
        run_id="r1",
        source_duration=25.0,
        decisions=[
            EditDecision(start=0.0, end=10.0, action="keep"),
            EditDecision(start=10.0, end=15.0, action="remove", reason="dead_air"),
            EditDecision(start=15.0, end=25.0, action="keep"),
        ],
        method="heuristic",
    )


# ---------------------------------------------------------------------------
# EditPlan validator
# ---------------------------------------------------------------------------

def test_editplan_accepts_contiguous_decisions():
    plan = _gap_plan()
    assert plan.clean_duration == 20.0
    assert plan.removed_seconds == 5.0


def test_editplan_rejects_gap():
    with pytest.raises(ValueError, match="contiguous"):
        EditPlan(
            run_id="r1", source_duration=20.0,
            decisions=[
                EditDecision(start=0.0, end=5.0, action="keep"),
                EditDecision(start=6.0, end=20.0, action="keep"),
            ],
        )


def test_editplan_rejects_overlap():
    with pytest.raises(ValueError, match="contiguous"):
        EditPlan(
            run_id="r1", source_duration=20.0,
            decisions=[
                EditDecision(start=0.0, end=12.0, action="keep"),
                EditDecision(start=10.0, end=20.0, action="keep"),
            ],
        )


def test_editplan_rejects_not_starting_at_zero():
    with pytest.raises(ValueError, match="start at 0"):
        EditPlan(
            run_id="r1", source_duration=20.0,
            decisions=[EditDecision(start=1.0, end=20.0, action="keep")],
        )


def test_editplan_rejects_not_ending_at_source_duration():
    with pytest.raises(ValueError, match="source_duration"):
        EditPlan(
            run_id="r1", source_duration=20.0,
            decisions=[EditDecision(start=0.0, end=15.0, action="keep")],
        )


def test_editplan_empty_decisions_requires_zero_duration():
    plan = EditPlan(run_id="r1", source_duration=0.0, decisions=[], method="identity")
    assert plan.clean_duration == 0.0
    with pytest.raises(ValueError, match="no decisions"):
        EditPlan(run_id="r1", source_duration=10.0, decisions=[], method="identity")


def test_editplan_normalizes_out_of_order_decisions():
    plan = EditPlan(
        run_id="r1", source_duration=20.0,
        decisions=[
            EditDecision(start=10.0, end=20.0, action="keep"),
            EditDecision(start=0.0, end=10.0, action="remove", reason="dead_air"),
        ],
    )
    assert [d.start for d in plan.decisions] == [0.0, 10.0]


def test_editplan_keep_ranges():
    plan = _gap_plan()
    assert plan.keep_ranges() == [(0.0, 10.0), (15.0, 25.0)]


# ---------------------------------------------------------------------------
# clean_to_source / source_to_clean -- identity plan is a true no-op
# ---------------------------------------------------------------------------

def test_identity_plan_clean_to_source_is_noop():
    plan = _identity_plan(30.0)
    for t in (0.0, 5.5, 29.9, 30.0):
        assert timeline.clean_to_source(plan, t) == pytest.approx(t)


def test_identity_plan_source_to_clean_is_noop():
    plan = _identity_plan(30.0)
    for t in (0.0, 5.5, 29.9, 30.0):
        assert timeline.source_to_clean(plan, t) == pytest.approx(t)


def test_clean_to_source_out_of_range_raises():
    plan = _identity_plan(10.0)
    with pytest.raises(ValueError):
        timeline.clean_to_source(plan, 11.0)
    with pytest.raises(ValueError):
        timeline.clean_to_source(plan, -1.0)


# ---------------------------------------------------------------------------
# Gap plan: the actual declutter case
# ---------------------------------------------------------------------------

def test_gap_plan_clean_to_source_before_and_after_cut():
    plan = _gap_plan()
    assert timeline.clean_to_source(plan, 5.0) == pytest.approx(5.0)   # before the cut
    assert timeline.clean_to_source(plan, 10.0) == pytest.approx(15.0)  # right at the cut boundary
    assert timeline.clean_to_source(plan, 15.0) == pytest.approx(20.0)  # after the cut


def test_gap_plan_source_to_clean_inside_removed_span_is_none():
    plan = _gap_plan()
    assert timeline.source_to_clean(plan, 12.0) is None
    assert timeline.source_to_clean(plan, 10.0) is not None  # exactly the removal's start (still "keep" side)


def test_gap_plan_round_trip_for_kept_content():
    plan = _gap_plan()
    for t_source in (0.0, 3.0, 9.9, 15.1, 20.0, 24.9):
        t_clean = timeline.source_to_clean(plan, t_source)
        assert t_clean is not None
        assert timeline.clean_to_source(plan, t_clean) == pytest.approx(t_source, abs=1e-6)


def test_source_ranges_for_spans_a_cut_boundary():
    plan = _gap_plan()
    # clip [5, 15) on the CLEAN timeline straddles the removed source gap
    ranges = timeline.source_ranges_for(plan, 5.0, 15.0)
    assert ranges == [(5.0, 10.0), (15.0, 20.0)]


def test_source_ranges_for_fully_within_one_keep_range():
    plan = _gap_plan()
    assert timeline.source_ranges_for(plan, 1.0, 4.0) == [(1.0, 4.0)]


def test_source_ranges_for_empty_when_end_before_start():
    plan = _gap_plan()
    assert timeline.source_ranges_for(plan, 10.0, 5.0) == []


# ---------------------------------------------------------------------------
# merge_small_gaps
# ---------------------------------------------------------------------------

def test_merge_small_gaps_noop_under_limit():
    ranges = [(0.0, 1.0), (2.0, 3.0)]
    assert timeline.merge_small_gaps(ranges, max_ranges=5) == ranges


def test_merge_small_gaps_merges_closest_pair_first():
    # gaps: (0-1)->(1.1-2) is 0.1; (1.1-2)->(5-6) is 3.0 -- the small gap merges first
    ranges = [(0.0, 1.0), (1.1, 2.0), (5.0, 6.0)]
    merged = timeline.merge_small_gaps(ranges, max_ranges=2)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_small_gaps_respects_cap():
    ranges = [(float(i), float(i) + 0.5) for i in range(0, 20, 1)]
    merged = timeline.merge_small_gaps(ranges, max_ranges=5)
    assert len(merged) <= 5


# ---------------------------------------------------------------------------
# apply_plan_to_transcript
# ---------------------------------------------------------------------------

def _transcript_with_gap_words() -> Transcript:
    # Words at 0-1, 2-3 (kept), 11-12 (inside the removed 10-15 gap), 16-17, 18-19 (kept)
    words = [
        Word(text="one", start=0.0, end=1.0),
        Word(text="two", start=2.0, end=3.0),
        Word(text="filler", start=11.0, end=12.0),
        Word(text="three", start=16.0, end=17.0),
        Word(text="four", start=18.0, end=19.0),
    ]
    seg = TranscriptSegment(id=0, start=0.0, end=19.0, text="one two filler three four", words=words)
    return Transcript(run_id="r1", source_path="x.mp4", duration=25.0, segments=[seg])


def test_apply_plan_to_transcript_drops_words_in_removed_span():
    plan = _gap_plan()
    t = _transcript_with_gap_words()
    cleaned = timeline.apply_plan_to_transcript(t, plan)

    all_words = cleaned.all_words()
    assert [w.text for w in all_words] == ["one", "two", "three", "four"]
    assert cleaned.duration == plan.clean_duration


def test_apply_plan_to_transcript_remaps_word_timestamps():
    plan = _gap_plan()
    t = _transcript_with_gap_words()
    cleaned = timeline.apply_plan_to_transcript(t, plan)

    words_by_text = {w.text: w for w in cleaned.all_words()}
    assert words_by_text["one"].start == pytest.approx(0.0)
    assert words_by_text["three"].start == pytest.approx(16.0 - 5.0)  # 5s removed before it


def test_apply_plan_to_transcript_drops_word_whose_span_exactly_equals_removal():
    # Regression: declutter.py's filler-word removal constructs the removed
    # EditDecision's start/end as EXACTLY the word's own start/end (see
    # _filler_word_removals). Both of that word's boundary points then land
    # exactly on the edge of an adjacent KEEP range, so checking start/end
    # independently (the old implementation) let both individually map
    # successfully -- producing a zero-duration "ghost" word instead of
    # dropping it. Caught by a real end-to-end run against real declutter
    # output; no prior fixture had a removal boundary exactly coincide with
    # a word boundary the way every real filler removal actually does.
    plan = EditPlan(
        run_id="r1", source_duration=10.0,
        decisions=[
            EditDecision(start=0.0, end=3.0, action="keep"),
            EditDecision(start=3.0, end=4.0, action="remove", reason="filler", text="um"),
            EditDecision(start=4.0, end=10.0, action="keep"),
        ],
    )
    words = [
        Word(text="hello", start=1.0, end=2.0),
        Word(text="um", start=3.0, end=4.0),  # spans EXACTLY the removed decision
        Word(text="world", start=5.0, end=6.0),
    ]
    seg = TranscriptSegment(id=0, start=1.0, end=6.0, text="hello um world", words=words)
    t = Transcript(run_id="r1", source_path="x.mp4", duration=10.0, segments=[seg])

    cleaned = timeline.apply_plan_to_transcript(t, plan)

    all_words = cleaned.all_words()
    assert [w.text for w in all_words] == ["hello", "world"]
    assert all(w.end > w.start for w in all_words)  # no zero-duration survivors


def test_apply_plan_to_transcript_identity_is_noop():
    plan = _identity_plan(25.0)
    t = _transcript_with_gap_words()
    cleaned = timeline.apply_plan_to_transcript(t, plan)

    assert [w.text for w in cleaned.all_words()] == [w.text for w in t.all_words()]
    assert cleaned.duration == t.duration


def test_apply_plan_to_transcript_drops_segment_with_no_surviving_words():
    plan = EditPlan(
        run_id="r1", source_duration=10.0,
        decisions=[EditDecision(start=0.0, end=10.0, action="remove", reason="manual")],
    )
    # An all-removed plan has no keep ranges at all -- clean_duration 0.
    assert plan.clean_duration == 0.0
    t = Transcript(
        run_id="r1", source_path="x.mp4", duration=10.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=5.0, text="gone", words=[Word(text="gone", start=0.0, end=5.0)])],
    )
    cleaned = timeline.apply_plan_to_transcript(t, plan)
    assert cleaned.segments == []


# ---------------------------------------------------------------------------
# Property-style check over a batch of random-ish plans (no hypothesis dep)
# ---------------------------------------------------------------------------

def test_clean_duration_always_equals_sum_of_keep_ranges():
    import random

    rng = random.Random(42)
    for _ in range(20):
        n_gaps = rng.randint(0, 4)
        cuts = sorted(rng.uniform(1, 99) for _ in range(n_gaps * 2))
        boundaries = [0.0] + cuts + [100.0]
        decisions = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end - start < 1e-9:
                continue
            action = "remove" if i % 2 == 1 and n_gaps else "keep"
            decisions.append(EditDecision(start=start, end=end, action=action, reason="manual" if action == "remove" else None))
        if not decisions:
            continue
        # Re-glue to guarantee contiguity after dropping any zero-length slivers.
        fixed = []
        cursor = 0.0
        for d in decisions:
            fixed.append(EditDecision(start=cursor, end=cursor + (d.end - d.start), action=d.action, reason=d.reason))
            cursor += d.end - d.start
        plan = EditPlan(run_id="r1", source_duration=cursor, decisions=fixed)
        expected = sum(d.end - d.start for d in fixed if d.action == "keep")
        assert plan.clean_duration == pytest.approx(expected, abs=5e-4)  # clean_duration rounds to 3 decimals
