# content-repurposer

## Purpose

Turns one selected `Clip` (plus its surrounding transcript text) into a
platform-native post for each requested platform. Backs
`pipeline/repurposer.py::generate_posts`, the same split as clip-selector:
this skill describes the LLM-enabled behavior, the Python module implements
it and a deterministic template fallback under the same contract.

## When invoked

Invoked automatically by `generate_posts()` **only when**
`pipeline.llm.get_llm_client()` returns a client. With no key, every post is
built by `pipeline/repurposer.py::_template_post` using the `PLATFORM_RULES`
table (char limits, hashtag counts, CTAs) — deterministic, same platforms
supported, no LLM call.

## Input schema

```json
{
  "clip": {
    "id": "clip-01", "hook": "the biggest mistake ...",
    "topic": "personal finance saving mistakes", "start": 41.2, "end": 87.9
  },
  "transcript_excerpt": "most people make a huge mistake with their first paycheck. the biggest mistake is spending everything before saving anything...",
  "platforms": ["linkedin", "x", "threads", "instagram", "shorts", "newsletter"]
}
```

## Output schema

```json
{
  "posts": [
    {
      "platform": "linkedin",
      "text": "Most people make one mistake with their first paycheck...",
      "hashtags": ["#personalfinance", "#saving"],
      "cta": "What's been your experience with this?"
    }
  ]
}
```
Parsed into `list[Post]` with `generation_method="llm"`, one entry per
requested platform.

## Prompting guidance (LLM path)

- System prompt: "You are a social media repurposing assistant. Given a
  video clip's hook and transcript text, write one platform-native post per
  requested platform. Respect each platform's conventions (length, tone,
  hashtag use)."
- Still worth constraining per-platform length even with an LLM — pass the
  same `PLATFORM_RULES.max_chars` figures from `pipeline/repurposer.py` in
  the prompt so LLM output and the template fallback stay comparable.
- Platforms not in `PLATFORM_RULES` (`linkedin`, `x`, `threads`,
  `instagram`, `shorts`, `newsletter`) are dropped by the parser.

## Fallback behavior (zero-key default)

`_template_post` in `pipeline/repurposer.py`, per platform:

| Platform | Shape | Max chars | Hashtags | CTA |
|---|---|---|---|---|
| linkedin | hook + body + CTA, paragraph breaks | 1200 | 4 | yes |
| x | hook + body, no CTA | 260 | 2 | no |
| threads | hook + body + CTA | 450 | 3 | yes |
| instagram | hook + body + CTA | 2000 | 8 | yes |
| shorts | `Title: ...` / `Description: ...` metadata | 480 | 3 | no |
| newsletter | `Subject: ...` + longer body | 2500 | 0 | yes |

Body text is reconstructed from the transcript segments the clip spans
(`Clip.source_segment_ids`). Hashtags are derived from `Clip.topic`, with
`Learnings.top_keywords` (once available) taking priority. All output is
deterministic and length-truncated on word boundaries — running the same
clip twice produces identical posts.
