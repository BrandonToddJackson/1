# content-analyst

## Purpose

Turns raw performance data (views/likes/comments/shares per post) into
`Learnings` — the winning keywords, hook patterns, platforms, and ideal
clip-length range from the batch that just ran. `Learnings` is written to
`learnings.json` at the repo root and picked up automatically by the next
`select-clips` and `repurpose` runs, closing the loop the architecture
diagram describes.

## When invoked

**Always local — there is no LLM path for this stage in v1.** Ranking
performance data and mining keywords from the winners is a statistics
problem, not a generation problem, so `pipeline/analyst.py::analyze` is the
only implementation; there's no LLM-enhanced variant to fall back from. An
LLM-assisted "explain *why* these won" summary is a plausible future
addition but is out of scope for v1 — see README.md's limitations section.

## Input schema

A CSV (or `list[PerformanceRecord]`) with one row per published post:

```csv
post_id,platform,clip_id,views,likes,comments,shares,engagement_rate
p-linkedin-01,linkedin,clip-01,5000,400,60,40,
```
`engagement_rate` is optional — computed as `(likes+comments+shares)/views`
when blank. Optionally also takes the run's `list[Post]` (for hashtags) and
`list[Clip]` (for hook/topic/duration) to enrich the analysis; without
them, `analyze()` still ranks platforms from the performance data alone.

## Output schema

```json
{
  "generated_at": "2026-08-28T12:00:00Z",
  "top_keywords": ["compoundinterest", "money", "mistake", "saving"],
  "top_hook_patterns": ["the biggest mistake with your first paycheck", "..."],
  "best_platforms": ["instagram", "linkedin", "threads", "x"],
  "ideal_clip_length_range": [28.5, 61.0],
  "notes": "Derived from 6 performance record(s); top 2 by engagement rate drove these learnings."
}
```
This is exactly `pipeline.schemas.Learnings` — written to `learnings.json`
by `pipeline/analyst.py::save_learnings`.

## Fallback behavior (always-on, no LLM)

`analyze()` in `pipeline/analyst.py`:

1. Computes `engagement_rate` per record, ranks all records, takes the top
   quartile (`top_fraction=0.25`, min 1).
2. `best_platforms`: every platform ranked by its *average* engagement rate
   across all records (not just the top quartile), so a platform with one
   viral fluke doesn't dominate.
3. `top_keywords`: hashtags from the top-quartile posts first (human-curated
   signal), then the most frequent non-stopword tokens mined from those
   clips' `hook`+`topic` text.
4. `top_hook_patterns`: the raw `hook` text of each top-quartile clip.
5. `ideal_clip_length_range`: min/max duration among the top-quartile clips.

`clip_selector.select_clips(..., learnings=...)` and
`repurposer.generate_posts(..., learnings=...)` both accept the resulting
`Learnings` object directly — `pipeline/cli.py` loads it automatically via
`analyst.load_learnings()` on every `select-clips`/`repurpose` call, so
running `analyze` once between batches is enough to feed the next run.
