# Motion philosophy

The style bible for every composition in `graphics/compositions/`. Read
this before adding a new composition to the catalog — consistency across
beats matters more than any single beat looking clever.

## Brand tokens

Every composition links `../tokens.css` and uses its custom properties
(`--gfx-bg`, `--gfx-fg`, `--gfx-accent`, `--gfx-radius`, `--gfx-font`,
`--gfx-ease`/`--gfx-ease-in`) rather than hard-coded colors/fonts. One
palette change in `tokens.css` should be enough to reskin the whole
catalog.

## Easing

- Entrances use `--gfx-ease` (`cubic-bezier(0.16, 1, 0.3, 1)`, a strong
  ease-out) — motion should feel like it's arriving with intent, not
  sliding in limply.
- Exits/emphasis-collapses use `--gfx-ease-in`. Never use linear easing for
  a card entrance; it reads as robotic.
- Keep entrance durations short: 250-400ms. A caption graphic is on screen
  for 1.2-2.5s total (see each catalog entry's fixed `duration`) — a slow
  entrance eats a third of that budget doing nothing.

## Safe areas

- Nothing meaningful inside the outer 6% margin on any edge (caption burn-in
  and platform UI chrome both live there — see `pipeline/qc.py`'s safe-area
  check, added in Step 9).
- `lower-third` and `facecam-frame` anchor to a corner/edge; `stat-card`,
  `quote-card`, and `keyword-pop` anchor center or center-low so they don't
  collide with karaoke/pop captions, which live bottom-center.

## Pacing — max beats per 30s

**No more than 3 graphics beats in any 30-second window of a clip.** A beat
every few seconds stops reading as emphasis and starts reading as noise —
this is the single most common way an automated-graphics pipeline looks
like slop rather than a deliberate edit. `graphics_max_beats` in
`pipeline/config.py` enforces a per-clip cap; a clip's LLM-proposed beat
list gets truncated to the highest-confidence beats if it exceeds this
budget, never silently over-dense.

## Duration is a catalog property, not a per-beat LLM output

Every composition's `duration` is fixed in `catalog.json`, not something
the LLM planner chooses per beat. This removes a whole failure class (a
hallucinated 47-second "stat card") and keeps the catalog's rhythm
consistent — every `stat-card` beat is the same length everywhere it's
used, so the pattern is legible to fast-scrolling viewers.

## Transparency contract

Every composition's `<html>`/`body` background is `transparent` (inherited
from `tokens.css`). Rendered via `hyperframes render --format mov` (ProRes
4444 with an alpha channel, confirmed via `ffprobe` — `pix_fmt=yuva444p12le`
against a real render in this repo's history). `pipeline/graphics.py`
composites the result over the captioned clip with ffmpeg's `overlay`
filter, gated to the beat's exact `[start, start+duration)` window via
`enable='between(t,S,S+D)'` — never touching the base video outside that
window.

## The LLM never emits HTML

The planner picks a `composition` id from the fixed catalog and fills its
declared `variables` (typed strings — see `catalog.json`). It never writes
markup, CSS, or timing math. This is the same validate-and-clamp safety
model this repo already applies to every other LLM output (clip selection,
declutter removals, post copy) — a hallucinated composition id or an
undeclared variable is dropped with a warning, not trusted blind, and
`hyperframes render --strict-variables` gives a second, independent
enforcement layer at render time.

## Anchor words, not timestamps

The LLM proposes an `anchor_word` (a short quote from the transcript), not
a timestamp — `pipeline/graphics.py::_snap_anchor` finds the nearest
matching `Word` and places the beat there. This is why graphics land on the
right word even though the LLM never sees exact float timestamps to
hallucinate.
