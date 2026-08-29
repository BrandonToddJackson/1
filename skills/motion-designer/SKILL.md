# motion-designer

## Purpose

Plans which motion-graphics compositions to overlay on a clip, and where.
This is the LLM-enabled half of `pipeline/graphics.py::plan_and_render_graphics`
-- the skill describes that behavior; the Python module implements it (when
both an LLM key AND a working local HyperFrames setup are present) and
degrades to a no-op (the clip passes through unchanged) otherwise.

## When invoked

Invoked automatically by `plan_and_render_graphics()` **only when**:
1. `pipeline.graphics.graphics_available()` returns `True` -- Node >= 22,
   `graphics/node_modules` installed (`npm install` has been run inside
   `graphics/`), and `graphics/catalog.json` present, **and**
2. `pipeline.llm.get_llm_client()` returns a configured client (i.e.
   `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set).

Missing either one skips this stage entirely: `GraphicsPlan(method="skipped",
skipped_reason=...)`, and the clip passes through to `final_clips.json`
byte-for-byte unchanged (still copied into its own file under
`clips_graphics/` -- see the project's governing rule: disabled is an
identity artifact, never a missing one).

## Input schema

The clip's own word-level transcript (a flat string of its words, not the
full `Transcript` object -- graphics only needs to find anchor phrases, not
segment boundaries) plus the composition catalog's ids/descriptions/
declared variables (see `graphics/catalog.json`).

## Output schema

```json
{
  "beats": [
    {
      "composition": "stat-card",
      "variables": {"value": "3.2x", "label": "faster onboarding"},
      "anchor_word": "grew three",
      "reason": "speaker states a specific growth multiple",
      "confidence": 0.9
    }
  ]
}
```

Parsed into `list[GraphicsBeat]`. Every entry is validated exactly like
every other LLM output in this repo:
- `composition` must be a real catalog id -- otherwise dropped, warned.
- `variables` must supply every variable that composition declares (as
  strings) -- missing any one drops the whole beat, warned. Unknown extra
  keys are silently ignored (the composition just won't read them).
- `anchor_word` (2-5 exact words quoted from the transcript) is snapped to
  the nearest matching `Word` via `pipeline/graphics.py::_snap_anchor` --
  no match at all drops the beat, warned. **The LLM never provides a
  timestamp** -- this is deliberate, see `graphics/MOTION.md`'s "Anchor
  words, not timestamps" section.
- `duration` is NOT an LLM output -- it's the catalog composition's own
  fixed `duration`, always. This removes a whole failure class (a
  hallucinated 47-second stat card) and keeps every use of a given
  composition the same length.
- If more beats survive validation than `graphics_max_beats` (default 4,
  see `pipeline/config.py`), the highest-`confidence` beats are kept, then
  re-sorted chronologically -- never a silent truncation to "whatever came
  first in the response."

## Prompting guidance (LLM path)

- System prompt lists the full catalog (id + description + declared
  variables) inline, and instructs: pick moments worth visual emphasis,
  one composition per beat, fill every declared variable, quote a short
  exact anchor phrase, at most `graphics_max_beats` beats total, **never
  more than 3 in any 30-second window** -- graphics are emphasis, not
  decoration; when in doubt, propose fewer, not more (see
  `graphics/MOTION.md`'s pacing rule).
- Force strict JSON output, same mechanism as `clip-selector`/`editor`.
- The LLM never writes HTML, CSS, or timing math -- only composition
  selection + typed variable values. `hyperframes render --strict-variables`
  is a second, independent enforcement layer at render time: a variable
  set that doesn't match the composition's own declared schema fails that
  one beat's render (dropped, the rest of the batch still composites),
  never silently renders garbage.

## The catalog (zero-key-safe, hand-authored)

Six compositions ship in `graphics/catalog.json` /
`graphics/compositions/`: `lower-third`, `stat-card`, `quote-card`,
`keyword-pop`, `facecam-frame`, `progress-dots`. Every one links
`graphics/tokens.css` for brand consistency and is rendered via
`npx hyperframes render --format mov` to a transparent ProRes 4444 MOV
(`pix_fmt=yuva444p12le`, confirmed against a real render in this repo's
history), then composited onto the clip with ffmpeg's `overlay` filter,
gated to the beat's exact time window. See `graphics/MOTION.md` for the
full style bible (easing, safe areas, pacing).

## Fallback behavior (no key, or no working local HyperFrames setup)

No heuristic fallback exists for graphics planning -- unlike clip
selection or declutter, picking a good visual-emphasis moment without any
model reasoning about it isn't something a keyword heuristic can credibly
do, so this stage is purely skip-or-LLM. `GraphicsPlan(method="skipped")`
is a complete, valid, successful result: the pipeline runs end-to-end with
zero keys and zero Node install, exactly like every other optional stage.
