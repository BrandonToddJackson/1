# content-strategist

## Purpose

Reads a full transcript and proposes *content opportunities* — themes,
angles, and rough time ranges worth turning into clips — before
clip-selector picks exact boundaries. Intended as a higher-level pass that
narrows and directs clip-selector, e.g. "this transcript has a strong
personal-finance angle around 2:00-4:00 and a contrarian take around
9:00-11:00; prioritize those."

## Scope note (v1)

**This is the one skill with no zero-key Python fallback module.** Unlike
clip-selector, content-repurposer, and content-analyst, there is no
`pipeline/content_strategist.py` — this stage requires an LLM to be useful
(topic/angle judgment isn't a good fit for a keyword heuristic the way clip
boundary-snapping and templated post copy are). Wiring it in is a
documented v1 scope reduction, not an oversight — see README.md's
limitations section.

**Behavior with no API key configured:** this stage is skipped entirely.
`clip_selector.select_clips()` runs directly against the raw transcript
using its own heuristic, exactly as if content-strategist had never run.
Nothing breaks; you just don't get the extra topic-level steer.

**Behavior once a key is configured:** wire this in ahead of clip-selector
in `pipeline/cli.py`'s `select-clips` step — call it first, and pass its
`content_opportunities` output as extra hints into `select_clips()` (e.g.
merged into `Learnings.top_keywords` for that run, or as explicit
time-range priorities). Not implemented in v1; the input/output contract
below is what a future `pipeline/content_strategist.py` (LLM-only, matching
the pattern in clip-selector's own `_select_clips_llm`) should target.

## Input schema

A `Transcript` (see `pipeline/schemas.py`), same shape clip-selector uses.

## Output schema

```json
{
  "opportunities": [
    {
      "topic": "the real cost of waiting to invest",
      "why_it_works": "concrete, contrarian, relatable to a broad audience",
      "suggested_platforms": ["linkedin", "instagram", "shorts"],
      "suggested_time_range": [41.0, 92.0]
    }
  ]
}
```

## Prompting guidance (LLM path)

- System prompt: "You analyze a long-form transcript and identify the
  strongest content opportunities for short-form repurposing. For each
  opportunity, name the topic, explain briefly why it would perform well,
  suggest which platforms fit it best, and give an approximate time range
  in the source."
- Keep this pass cheap — segment-level transcript only (no word timestamps),
  and cap opportunities at ~5-8 so it doesn't just restate every sentence.
- Downstream: `clip_selector` still owns exact clip boundaries and scoring;
  this stage only narrows *where* it should look.
