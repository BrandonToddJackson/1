# editor

## Purpose

Finds retakes, false starts, and verbatim repetitions in a timestamped
transcript so they can be cut before the speaker's best take reaches
viewers. This is the LLM-enabled half of `pipeline/declutter.py::declutter`
— the skill describes that behavior; the Python module implements it (when
a key is configured) and unions it with a deterministic zero-key heuristic
pass (filler-word and dead-air removal) that runs regardless.

## When invoked

Invoked automatically by `declutter()` **only when**
`pipeline.llm.get_llm_client()` returns a configured client (i.e.
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set) and the requested
`level` isn't `"off"`. With no key, only the heuristic pass runs
(`pipeline/declutter.py::_heuristic_removals`) — filler words and dead air,
never retakes (a heuristic can't tell a deliberate repeat from a retake).

## Input schema

A `Transcript` (see `pipeline/schemas.py`), rendered for the model as
timestamped segment lines — the same shape `clip-selector` uses:

```json
{
  "run_id": "20260828-120000-ab12cd",
  "duration": 612.4,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "so the the biggest mistake -- sorry, let me start over. the biggest mistake people make..."}
  ]
}
```

## Output schema

```json
{
  "removals": [
    {"start": 1.8, "end": 6.1, "reason": "false_start", "text": "so the the biggest mistake -- sorry, let me start over."}
  ]
}
```

Parsed into removal spans and unioned with the heuristic pass's spans
(`pipeline/declutter.py::_merge_spans`), then built into an `EditPlan`
(see `pipeline/schemas.py`). `reason` must be one of `"retake"`,
`"false_start"`, `"repetition"` — anything else defaults to `"retake"`.
Every entry is clamped to the transcript's own duration and validated
exactly like `clip-selector`'s LLM output (a malformed or out-of-range
entry is dropped with a warning, never trusted blind).

## Prompting guidance (LLM path)

- System prompt: "You find retakes, false starts, and verbatim repetitions
  in a timestamped transcript of someone recording themselves talking to
  camera. Return ONLY spans that should be CUT (removed) -- never spans to
  keep. A retake is when the speaker restarts a sentence or thought; keep
  only the final, best version and mark the earlier attempt(s) for removal.
  ... Be conservative: when in doubt, don't mark it."
- Force strict JSON output, same mechanism as `clip-selector`.
- Give the model the full segment-level transcript (not word-level -- word
  timestamps are only needed downstream, once a removal span's exact word
  boundaries are known via `pipeline/timeline.py`).
- **Bias toward under-removal.** A missed retake is a minor annoyance a
  viewer scrolls past; an over-eager removal that cuts real content is a
  correctness bug. The prompt says "be conservative" for this reason, and
  the code backs it with a hard guard (next section) rather than trusting
  the prompt alone.

## The 40% rejection guard

`declutter()` sums the LLM's proposed removals and discards the entire LLM
result (falling back to heuristic-only, not partial) if it would remove
more than `LLM_MAX_REMOVAL_FRACTION` (40%) of total duration. This is a
defensive guard against a model that inverts keep/remove semantics or
otherwise goes badly wrong — a single bad response should never be able to
silently gut most of a video. The rejection is logged as a warning; the run
still succeeds using the heuristic plan.

## Fallback behavior (zero-key default)

`_heuristic_removals` in `pipeline/declutter.py` — see that module's
docstring for the full filler/dead-air ladder (`level` = light/standard/
aggressive). It never attempts retake/repetition detection; that is LLM-only
by design, since spotting "the speaker said this twice and meant the second
one" requires actual comprehension a keyword/pause heuristic cannot fake.

Both paths return an `EditPlan` with `method` set to `"llm"` (when the LLM
contributed at least one surviving removal) or `"heuristic"` (identity or
heuristic-only) so downstream stages and QC checks can tell which one ran.
