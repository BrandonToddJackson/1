# Content Pipeline

A long-form video/podcast → best clips → platform-specific posts → publish
→ learn pipeline that also does real automated editing: denoise/normalize,
filler/dead-air removal, karaoke/pop captions, speaker diarization, and
LLM-planned motion graphics — the quality bar of an automated
Descript/Riverside replacement, not a stripped-down subset of one.
**Runs end-to-end with zero API keys and zero paid dependencies** — every
stage has a free/local default; LLM and diarization/publishing/graphics
providers are strictly opt-in via `.env`.

```
[yt-dlp / local file] → INGEST
   → [ffmpeg: denoise, de-ess, normalize] ENHANCE
   → [faster-whisper, CPU | ElevenLabs Scribe | +pyannote diarization] TRANSCRIBE
   → DECLUTTER (filler/dead-air heuristic, +LLM retake detection) → clean transcript + EditPlan
   → CLIP-SELECTOR (heuristic default | LLM if key present)
   → [ffmpeg filter_complex EDL] CUT
   → [ffmpeg burn-in: plain SRT | karaoke/pop ASS] CAPTION
   → GRAPHICS (LLM-planned HyperFrames overlays, skipped without Node+key)
   → CONTENT-REPURPOSER (template default | LLM if key) → POSTS
   → PUBLISHER (local outbox/ | live Blotato/Ayrshare if key present)
   → performance.csv (you supply) → CONTENT-ANALYST → learnings.json
   → feeds back into CLIP-SELECTOR + REPURPOSER on the next run

   (non-stage commands, run any time after `caption`/`graphics`:)
   → PREVIEW (extract PNG frames for visual review)
   → QC (objective, deterministic checks -- duration/loudness/silence/captions/safe-area)
```

`skills/*.md` = the brain (LLM-enabled contracts). `pipeline/*.py` = the
hands + nervous system (deterministic JSON in, JSON out at every stage —
see `pipeline/schemas.py`). No orchestration platform required; any stage
is a pure function you can lift into Pipedream/Activepieces later.

## Quick start

```bash
# 1. System dependency (not pip-installable):
apt-get install -y ffmpeg

# 2. Python dependencies:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the whole pipeline on a local file or a URL yt-dlp supports:
python -m pipeline.cli run path/to/video.mp4 --max-clips 5

# Inspect the result:
ls runs/<run_id>/outbox/          # per-clip, per-platform post drafts + media
```

No `.env` file is required for this. The **first** `run` downloads the
`faster-whisper` "base" model (~150MB, cached under `~/.cache` afterwards) —
that's the only network dependency beyond an optional `yt-dlp` download.
`enhance`, `declutter` (default off, see below), `cut`, and `caption` are
pure `ffmpeg` — no other downloads. `graphics` needs a one-time Node setup
(see its own section below) and is skipped cleanly without it.

Each stage is also runnable on its own — useful for debugging, or resuming
after an interruption:

```bash
python -m pipeline.cli ingest path/to/video.mp4        # prints a run_id
python -m pipeline.cli enhance <run_id>                 # --off to skip
python -m pipeline.cli transcribe <run_id>
python -m pipeline.cli declutter <run_id>                # --level off|light|standard|aggressive
python -m pipeline.cli select-clips <run_id> --max-clips 5
python -m pipeline.cli cut <run_id>
python -m pipeline.cli caption <run_id>                  # --style plain|karaoke|pop
python -m pipeline.cli graphics <run_id>                 # --only <clip-id> to redo just one
python -m pipeline.cli repurpose <run_id> --platforms linkedin,x,shorts
python -m pipeline.cli publish <run_id>
python -m pipeline.cli analyze performance.csv --run-id <run_id>
python -m pipeline.cli preview <run_id>                   # PNG frames for visual review
python -m pipeline.cli qc <run_id>                        # objective checks -> qc.json
```

`run --run-id <run_id>` re-invoked with the same id **resumes**: stages
already recorded in `runs/<run_id>/run.json`'s `stages_completed` are
skipped, so a stalled download or a flaky whisper run doesn't cost you the
whole batch. If a stage genuinely failed, `PipelineRun.status` is
`"failed"` with a real error message (`run.json`'s `error` field) — just
re-run the same command and it picks up where it left off, never
re-downloading or re-transcribing work that already succeeded.

**Re-running a stage discards everything downstream of it.** `select-clips`
run again on an already-`cut`/`caption`/`publish`-ed run deletes those
stages' output before producing new clips — a stale cut video can never
survive a re-selection with different parameters. `cut`/`caption`/`graphics`
themselves are the exception: a retry after a partial `ffmpeg`/render
failure resumes by skipping clips whose output already exists, rather than
redoing the whole batch.

Re-running `run` with **different tuning params** than a previous run of
the same `--run-id` used refuses by default, naming what changed:

```
$ python -m pipeline.cli run video.mp4 --run-id abc123 --max-clips 8
refusing: run_id=abc123 was previously run with different params
(max_clips changed). Pass --force to redo from the earliest affected
stage, or omit the changed options to resume as-is.
```

Pass `--force` to redo from the earliest stage that param actually affects
— changing only `--platforms`, for instance, re-runs `repurpose`/`publish`
but leaves your cuts, captions, and graphics untouched. An unchanged
re-run of an already-complete `run_id` is a clean no-op ("nothing to do").

## The governing rule: every stage always runs, always writes its artifact

`enhance` disabled (`--no-enhance`) still writes `enhanced_media.json` —
pointing at the untouched source. `declutter` at its default `--level off`
still writes `edit_plan.json` (an identity plan: one "keep" decision
spanning the whole source) and a byte-identical `transcript_clean.json`.
`graphics` skipped (no LLM key, or Node/`graphics/node_modules` missing)
still writes `graphics_plans.json` (`method="skipped"`) and a
`final_clips.json` entry pointing at a straight copy of the captioned clip.
**Disabled means an identity artifact, never a missing one** — every
downstream stage always has something valid to read, and the CLI's
resumability/invalidation machinery (`STAGE_REQUIRES`/`PARAM_AFFECTS` in
`cli.py`) never needs to special-case "did this optional stage actually
run."

## Declutter defaults to `--level off` — here's why

`select-clips`/`cut`/`caption`/`repurpose` all read `transcript_clean.json`
(declutter's output), and `clip.start`/`clip.end` are **clean-timeline**
coordinates from the moment `declutter` actually removes anything.
`cutter.cut_ranges()` (used by `cut`) is already fully EDL-aware — it maps
those clean-timeline clip boundaries back to one-or-more source-timeline
ranges via `pipeline/timeline.py::source_ranges_for`, so turning declutter
on is safe and correct today. The `off` default is a deliberate product
choice, not a limitation: filler/dead-air removal changes what viewers
hear, and that's a call worth making on purpose per-project rather than
silently on every run. Turn it on with `--declutter-level light` (hard
fillers only), `standard` (+ flanked soft fillers, the recommended
starting point), or `aggressive` (+ all soft fillers) — see
`pipeline/declutter.py`'s module docstring for the full ladder.

## Adding API keys later

Copy `.env.example` to `.env` and set any of:

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — `select-clips`, `declutter`
  (retake/false-start detection, unioned with its heuristic pass),
  `graphics` (composition/variable planning), and `repurpose` switch from
  the heuristic/template path to an LLM-generated one automatically
  (Anthropic preferred if both are set). See `pipeline/llm.py::get_llm_client()`
  — the single seam every LLM-optional stage checks. A bad/unparseable LLM
  response (markdown fences, a refusal, hallucinated timestamps, an
  undeclared composition variable) is validated and, if it can't be
  salvaged, falls back to the heuristic/template/skipped path with a
  logged warning rather than crashing the run.
- `ELEVENLABS_API_KEY` — `transcribe` uses ElevenLabs Scribe instead of
  local `faster-whisper`, bundling diarization + non-speech audio-event
  tags (laughter, applause) into the one call. **UNVERIFIED**: the
  request/response shape was reconstructed from ElevenLabs' Speech-to-Text
  documentation, not confirmed against a live key (`api.elevenlabs.io` was
  egress-blocked during implementation, the same honest caveat this repo
  already carries for Blotato below) — verify against a real key before
  relying on it in production.
- `HUGGINGFACE_TOKEN` — (only matters if `ELEVENLABS_API_KEY` isn't set)
  adds a `pyannote.audio` diarization overlay on top of the local
  `faster-whisper` transcript. Free, but gated — create a token at
  huggingface.co and accept `pyannote/speaker-diarization-3.1`'s license
  first. Requires `pip install -e ".[diarize]"` (pulls in `torch`,
  large). `TRANSCRIBE_BACKEND`/`DIARIZATION_BACKEND` (both default
  `auto`) let you force a tier explicitly — see
  `pipeline/transcribe.py::get_transcribe_backend()`.
- `BLOTATO_API_KEY` + `BLOTATO_ACCOUNT_IDS` (a JSON object, one entry per
  platform you want to post to, e.g. `{"x": "acct_123", "linkedin":
  "acct_456"}` — get the ids from your Blotato dashboard's connected
  accounts) — `publish` attempts a live post via Blotato instead of writing
  a local outbox folder. Blotato is preferred over Ayrshare if both keys
  are set (cheaper, with a real REST API). **UNVERIFIED** for the same
  documentation-only reason as ElevenLabs above.
- `AYRSHARE_API_KEY` — same idea, used if `BLOTATO_API_KEY` isn't set.

Either live-publish path needs `pip install -e ".[publish]"` (adds the
`requests` package, not installed by default) — without it, `publish`
reports a clean per-post failure rather than crashing. **Neither provider
can receive your local clip video** — both need a publicly-hosted media
URL, so live publish is text-only; anything needing the actual video still
goes through the outbox for manual upload. `newsletter` has no live target
on either provider and always routes to the outbox regardless of which key
is set.

Nothing else changes. The output shapes (`Clip`, `Post`, `PublishResult`,
`EditPlan`, `GraphicsPlan`) are identical either way.

## Setting up `graphics` (motion-graphics overlays via HyperFrames)

Scoped entirely to its own `graphics/` directory — invisible to the rest of
the repo if you never run this stage:

```bash
cd graphics
npm install          # pulls the pinned `hyperframes` package (0.8.17)
cd ..
```

`hyperframes render` needs a local Chrome — it downloads
`chrome-headless-shell` on first use (run `npx hyperframes browser ensure`
inside `graphics/` to trigger that up front rather than mid-render). Then
set an LLM key (above) and `graphics` runs for real; without either piece
it's cleanly `method="skipped"` (see the governing-rule section above), not
an error.

The catalog (`graphics/catalog.json` + `graphics/compositions/*.html`)
ships 6 hand-authored compositions — `lower-third`, `stat-card`,
`quote-card`, `keyword-pop`, `facecam-frame`, `progress-dots` — every one
verified end-to-end (`hyperframes render --format mov --strict
--strict-variables`, real ProRes-4444-with-alpha output confirmed via
`ffprobe`) during implementation on this exact toolchain. See
`graphics/MOTION.md` for the style bible (easing, safe areas, the "never
more than 3 beats per 30s" pacing rule) and
`skills/motion-designer/SKILL.md` for the LLM planning contract. The LLM
never writes HTML — it only picks a composition id and fills its declared,
typed variables; `hyperframes render --strict-variables` is a second,
independent enforcement layer at render time.

### The subjective loop: `preview` → hand-edit → `graphics --only` → `preview`

There's no autonomous `--iterate N` render/critique/re-render loop in this
CLI, on purpose — that would be non-deterministic and unbounded in cost,
breaking the one invariant the whole resumability model exists to protect
(same `run_id` + same params → same output). Instead:

1. `python -m pipeline.cli preview <run_id>` — extracts PNG frames per clip
   (5 evenly-spaced + the midpoint of every graphics beat, or pass
   `--at t1,t2,...` for exact timestamps) under `runs/<run_id>/preview/`.
2. Look at the PNGs (a human, or Claude Code reading them directly).
3. Not right? Hand-edit `runs/<run_id>/graphics_plans.json` — it's a
   validated, hand-editable `list[GraphicsPlan]`.
4. `python -m pipeline.cli graphics <run_id> --only <clip-id>` re-renders
   just that one clip's beats (deterministic render caching means an
   unchanged beat's already-rendered alpha MOV is reused, not re-rendered).
5. `preview` again.

This gives Herk's actual iterate-on-the-look loop with judgment where it
belongs (a human or an agent, interruptible) and reproducibility where it
belongs (the pipeline). `qc` (next section) is the fully-automated
complement: objective checks that need no human judgment at all.

## `qc`: objective, deterministic checks (not a substitute for `preview`)

```bash
python -m pipeline.cli qc <run_id>
```

Writes `runs/<run_id>/qc.json` (`QCReport`, see `pipeline/qc.py`) — every
check is measurable and reproducible, never "does this look good":

- **duration** — measured clip duration vs. its expected (`clip.end -
  clip.start`) length; catches EDL math drift in `cut`.
- **loudness** — measured integrated LUFS vs. `AUDIO_TARGET_LUFS`; catches
  an `enhance` regression or a re-encode that drifted.
- **silence** — any silence run in the final clip longer than
  `declutter.DEAD_AIR_THRESHOLD` that survived anyway (declutter was off,
  or missed it).
- **caption_cue_length** — a caption cue on screen longer than
  `qc.MAX_CUE_SECONDS` (4s) — hard to read on a fast-scrolling platform.
- **graphics_safe_area** (info-level, not a warning) — a bottom-anchored
  graphics beat (`lower-third`, `progress-dots`) that may visually collide
  with burned-in captions, which also live bottom-center.

## Stage-to-vendor mapping (if you want to swap in the original stack)

| Stage | v1 (zero-key) | Original paid-vendor idea | Swap-in note |
|---|---|---|---|
| Orchestration | local Typer CLI | Pipedream / Activepieces | every `pipeline/*.py` function is pure JSON-in/JSON-out — lift any stage into a Pipedream step directly |
| Ingest | `yt-dlp` + local file | Pipedream trigger/webhook | already free, no change needed |
| Enhance | `ffmpeg` (afftdn/deesser/speechnorm/loudnorm) | Descript Studio Sound | `RNNOISE_MODEL_PATH` upgrades denoise to `arnndn` if you supply a `.rnnn` model file |
| Transcribe | `faster-whisper` (CPU) | ElevenLabs Scribe / WhisperX | set `ELEVENLABS_API_KEY` (bundles diarization) or `HUGGINGFACE_TOKEN` (adds `pyannote` overlay) — same `Transcript` output shape either way |
| Declutter | heuristic filler/dead-air + optional LLM retakes | Descript Studio filler-word removal | `--declutter-level`; the LLM path is unioned with the heuristic one, never a replacement |
| Clip selection | heuristic scorer | GPT-5.x / Claude via a clip-selector skill | set `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` — `clip_selector.py` already has the LLM path built in |
| Cut | `ffmpeg` filter_complex EDL | `ffmpeg` / Auto-Editor | as-is; `cutter.cut_ranges()` already handles multi-segment EDLs |
| Caption | `ffmpeg` burn-in (plain/karaoke/pop) | Descript/CapCut auto-captions, Remotion | `--style karaoke`/`pop` for word-level ASS captions with no new dependency; `captioner.py`'s interface is narrow on purpose for a future Remotion swap |
| Graphics | HyperFrames (local, LLM-planned) | Descript/Canva motion templates | scoped to `graphics/`; add compositions to `graphics/catalog.json` |
| Generate posts | templates | GPT / Claude via content-repurposer skill | same seam as clip selection |
| Publish | local outbox | Blotato (preferred) / Ayrshare / Postiz | set `BLOTATO_API_KEY`+`BLOTATO_ACCOUNT_IDS` or `AYRSHARE_API_KEY`; a Postiz adapter would live alongside `_publish_blotato`/`_publish_ayrshare` in `publisher.py`, sharing `build_caption()` |
| Measure/learn | CSV + local stats | DB/Sheet + LLM | `analyst.py` is intentionally always-local/statistical; an LLM "why did these win" summary layer could sit on top of `Learnings` without changing its shape |

## Limitations

- **`declutter` defaults to `--level off`** — a deliberate product choice
  (see its own section above), not a technical limitation; `cut` is fully
  EDL-aware and turning it on is safe.
- **ElevenLabs and Blotato are UNVERIFIED against a live key** — both were
  reconstructed from documentation search snippets because their docs
  hosts were egress-blocked during implementation. Confirmed correct:
  request shape, headers, response parsing logic (validated against
  hand-built fixture JSON matching the documented shape). Not confirmed: a
  real network round-trip. Verify before relying on either in production.
- **HyperFrames was verified for real, on this box** — `npm install`
  against the real npm registry, and a real `chrome-headless-shell`
  render producing genuine ProRes-4444-alpha output, confirmed via
  `ffprobe`, during implementation. A different machine without a
  pre-cached Chrome download will need `npx hyperframes browser ensure`
  (or the first real render) to fetch it — that step itself wasn't
  re-verified from a cold cache here.
- **No live publishing without a key, and no video even with one** —
  `publish` writes a local `outbox/` folder unless a live key is
  configured; even then, neither Blotato nor Ayrshare can receive your
  local clip files (both need a publicly-hosted media URL), so live
  publish is text-only.
- **content-strategist has no offline fallback** — it's the one skill
  that's purely LLM-enhanced (see `skills/content-strategist/SKILL.md`).
  Without a key it's skipped; `clip_selector`'s heuristic runs standalone.
- **Heuristic clip scoring is simpler than an LLM's judgment** — it works
  well on transcripts with clear hook language and pauses, less well on
  monotone or jargon-heavy source audio. `analyst.py`'s `learnings.json`
  feedback loop is what's meant to close that gap over successive batches.
- **`yt-dlp` ingest is exercised in tests only at the CLI/argument level** —
  real YouTube downloads depend on external network conditions outside
  this repo's control; point `run` at a local file if you want a fully
  reproducible test run.
- **Performance CSVs must reference the run they came from** — `Clip` ids
  are scoped to the run that generated them (`<run_id>-clip-01`), so
  `analyze --run-id` needs to point at the same run the CSV's `clip_id`
  values actually came from. Pointing it at the wrong run doesn't error —
  `analyst.analyze()` adds a note to `Learnings.notes` explaining the
  mismatch instead of silently producing weaker learnings.
- **Performance CSV parsing is tolerant but not infinitely so** —
  `post_id`/`platform` are required columns; `views`/`likes`/`comments`/
  `shares`/`engagement_rate` handle thousands separators, float-style
  values, a trailing `%`, and Excel/Sheets BOMs, but a genuinely
  unparseable value in a present column raises a clear error naming the
  file/row/column rather than silently defaulting to 0.
- **`qc`'s loudness check has a known synthetic-audio blind spot** — a
  pure single-frequency test tone (as used in this repo's own test
  fixtures) has near-zero dynamic range, a documented `loudnorm` edge case
  where ffmpeg can't reliably hit a target — real speech doesn't have this
  problem.

## Testing

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

399 tests, all passing with **zero external services and zero API keys**
(pyflakes clean):

- **Pure Python** (no binaries needed): `test_schemas.py`, `test_storage.py`,
  `test_config.py`, `test_llm_fallback.py`, `test_llm_json.py`,
  `test_textutils.py`, `test_timeline.py` (source↔clean timeline mapping +
  a property test over random `EditPlan`s), `test_declutter.py` (the
  filler/dead-air heuristic ladder + the LLM retake path + its 40%-removal
  rejection guard), `test_captioner_ass.py` (drift-free `\k` karaoke
  timing, brute-force regression-tested the same way `_format_srt_time`
  already was), `test_transcribe_backend.py` (the ElevenLabs/pyannote
  precedence ladder and every fallback chain, monkeypatched, no real
  network/pyannote install), `test_graphics_plan.py` (catalog loading
  against the REAL `graphics/catalog.json`, anchor-word snapping, the LLM
  planner's full validation/capping logic, the skip ladder), `test_qc.py`,
  `test_preview.py`.
- `test_clip_selector.py`, `test_repurposer.py`, `test_analyst.py`,
  `test_publisher.py` — heuristic/template/statistical logic against
  fixture data in `fixtures/`, including LLM-path fallback/validation, the
  Blotato/Ayrshare payload shapes, and CSV-parsing edge cases.
- **Real ffmpeg against synthetic lavfi media** (no network, no checked-in
  fixtures): `test_media.py`, `test_procutil.py`, `test_audio_media.py`
  (measured loudness lands near target, video stream untouched byte-for-
  byte), `test_cutter_edl.py` (multi-range EDL cutting: duration-is-sum-
  of-parts, no-audio-source, the range cap), `test_graphics_composite.py`
  (the ffmpeg overlay/compositing logic against a self-generated qtrle
  alpha MOV — no Node needed), `test_qc_media.py`, `test_preview_media.py`.
- **Real HyperFrames** (gated on `graphics_available()` — skips, never
  fails, wherever Node/`graphics/node_modules` aren't present):
  `test_graphics_media.py` renders an actual catalog composition through
  the real `npx hyperframes render`, confirms genuine alpha-channel
  output, deterministic render caching, and the full plan→render→composite
  pipeline against a real clip.
- `test_cli.py`, `test_cli_resume.py`, `test_cli_preview_qc.py` — drive
  the full `run` command end-to-end (all 10 stages) against synthetic
  video, with only `transcribe` stubbed (see below). `test_cli_resume.py`
  covers the resumability model across every stage: failed-stage resume,
  invalidation cascades, `--force`/param-mismatch handling, and partial-
  failure retry-by-skipping.

### Manual end-to-end smoke test (with real transcription)

`transcribe.py`'s local path's only external dependency is downloading the
`faster-whisper` model weights from Hugging Face on first use — everything
else in this repo (including this test suite) works fully offline after
`pip install`. To verify transcription for real:

```bash
# Generate a short spoken sample (or use any real recording you have):
apt-get install -y espeak-ng
espeak-ng -f script.txt -w speech.wav -s 150
ffmpeg -f lavfi -i "color=c=navy:size=640x360:rate=15:duration=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 speech.wav)" \
       -i speech.wav -c:v libx264 -c:a aac -shortest sample.mp4

# Run the full pipeline with the small "tiny" model for speed:
WHISPER_MODEL=tiny python -m pipeline.cli run sample.mp4 --max-clips 2
```

This build session's own network policy blocks `huggingface.co`, so the
faster-whisper model download itself couldn't be completed here —
everything downstream of transcription (`ffmpeg` enhance/cut/caption,
HyperFrames rendering, template/heuristic logic, the full CLI wiring and
resume behavior) *was* verified against real binaries and multiple real
end-to-end `run`s. Run the command above on your own machine to confirm
transcription too. The same applies to a real ElevenLabs key
(`api.elevenlabs.io` was also blocked) and a real `pyannote.audio` run
against a genuine multi-speaker recording (two `espeak-ng` voices
concatenated makes a reasonable synthetic two-speaker test file) — both
should just work with normal internet access and the relevant key/token.

## Project layout

```
pipeline/            all stage modules + cli.py (see module docstrings);
                       procutil.py (subprocess error handling) and
                       textutils.py (shared text truncation) are small
                       helpers used across the media and copy-generation
                       stages respectively; timeline.py is the pure
                       source<->clean-timeline mapping declutter/cut share
graphics/             HyperFrames Node project (scoped, own package.json) --
                       catalog.json + compositions/*.html + tokens.css +
                       MOTION.md; node_modules/ and renders/ are gitignored
skills/               the 6 SKILL.md docs (LLM-enabled contracts)
fixtures/              sample transcript + performance CSV used by tests
tests/                  pytest suite (see Testing above)
runs/                    gitignored — per-run output (created on first use)
learnings.json           gitignored — written by `analyze`, read by
                          select-clips/repurpose on every subsequent run
```
