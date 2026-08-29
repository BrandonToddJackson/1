# clip-selector

## Purpose

Picks the best short-form clips out of a long-form timestamped transcript.
This is the "brain" behind `pipeline/clip_selector.py::select_clips` — the
skill describes the LLM-enabled behavior; the Python module implements both
that behavior (when a key is configured) and a deterministic zero-key
fallback that follows the same contract.

## When invoked

Invoked automatically by `select_clips()` **only when**
`pipeline.llm.get_llm_client()` returns a configured client (i.e.
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set). With no key, the pipeline
uses the deterministic fallback implemented in
`pipeline/clip_selector.py::_select_clips_heuristic` instead — same input,
same output shape, no LLM call.

## Input schema

A `Transcript` (see `pipeline/schemas.py`), rendered for the model as
timestamped lines:

```json
{
  "run_id": "20260828-120000-ab12cd",
  "source_path": "runs/20260828-120000-ab12cd/source.mp4",
  "language": "en",
  "duration": 612.4,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "welcome back to the show...", "words": [...]}
  ]
}
```

## Output schema

```json
{
  "clips": [
    {
      "start": 41.2,
      "end": 87.9,
      "hook": "the biggest mistake people make with their first paycheck",
      "topic": "personal finance saving mistakes",
      "score": 0.87,
      "caption_hint": "punchy, slight urgency"
    }
  ]
}
```
Parsed into `list[Clip]` with `selection_method="llm"`. Non-overlapping,
ordered by start time, each clip's duration within the caller's
`min_len`/`max_len` bounds (defaults: 20s-90s).

## Prompting guidance (LLM path)

- System prompt: "You select the best short-form clips from a timestamped
  transcript for social media. Pick clips with a strong hook, a complete
  thought, and a duration between {min_len} and {max_len} seconds. Return at
  most {max_clips} clips, non-overlapping, ordered by start time."
- Force strict JSON output (`response_format={"type": "json_object"}` for
  OpenAI; explicit "respond ONLY with valid JSON" instruction for Anthropic).
- Give the model the full segment-level transcript (not word-level — too
  many tokens); word-level timestamps are only needed downstream, by the
  captioner, once a clip's segment IDs are known.

## Fallback behavior (zero-key default)

`_select_clips_heuristic` in `pipeline/clip_selector.py`:

1. Finds candidate boundaries at pauses (word-to-word gap ≥ 0.35s) and
   sentence-ending punctuation in the word-level timestamps.
2. Builds every valid window between two boundaries whose length falls in
   `[min_len, max_len]`.
3. Scores each window: hook-word density (against a built-in vocabulary of
   generic attention words, extended by `Learnings.top_keywords` once a
   content-analyst run has produced real data) + closeness to an ideal
   length + a bonus if the window opens on a hook word.
4. Greedily selects the highest-scoring non-overlapping windows (non-max
   suppression) up to `max_clips`, then re-sorts chronologically.
5. `hook` = first ~12 words of the window; `topic` = most frequent
   non-stopword tokens in the window.

Both paths return `Clip` objects with `selection_method` set to `"llm"` or
`"heuristic"` so downstream stages (and analytics) can tell which one ran.
