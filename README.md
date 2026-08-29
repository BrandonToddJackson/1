# Content Pipeline

A long-form video/podcast → best clips → platform-specific posts →
publish → learn pipeline. **Runs end-to-end with zero API keys** — local
heuristics and templates by default, with clean seams to drop in paid LLM
and publishing APIs later.

```
[yt-dlp / local file] → INGEST → [faster-whisper, CPU] → TRANSCRIBE
   → CLIP-SELECTOR (heuristic default | LLM if key present) → [ffmpeg] CUT
   → [ffmpeg burn-in captions] CAPTION
   → CONTENT-REPURPOSER (template default | LLM if key) → POSTS
   → PUBLISHER (local outbox/ | live Ayrshare if key present)
   → performance.csv (you supply) → CONTENT-ANALYST → learnings.json
   → feeds back into CLIP-SELECTOR + REPURPOSER on the next run
```

`skills/*.md` = the brain (LLM-enabled contracts). `pipeline/*.py` = the
hands + nervous system (deterministic JSON in, JSON out at every stage —
see `pipeline/schemas.py`). No orchestration platform required for v1; any
stage is a pure function you can lift into Pipedream/Activepieces later.

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

Each stage is also runnable on its own — useful for debugging, or resuming
after an interruption:

```bash
python -m pipeline.cli ingest path/to/video.mp4        # prints a run_id
python -m pipeline.cli transcribe <run_id>
python -m pipeline.cli select-clips <run_id> --max-clips 5
python -m pipeline.cli cut <run_id>
python -m pipeline.cli caption <run_id>
python -m pipeline.cli repurpose <run_id> --platforms linkedin,x,shorts
python -m pipeline.cli publish <run_id>
python -m pipeline.cli analyze performance.csv --run-id <run_id>
```

`run --run-id <run_id>` re-invoked with the same id **resumes**: stages
already recorded in `runs/<run_id>/run.json`'s `stages_completed` are
skipped, so a stalled download or a flaky whisper run doesn't cost you the
whole batch.

## Adding API keys later

Copy `.env.example` to `.env` and set any of:

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — `select-clips` and `repurpose`
  switch from the heuristic/template path to an LLM-generated one
  automatically (Anthropic preferred if both are set). See
  `pipeline/llm.py::get_llm_client()` — this is the single seam every
  LLM-optional stage checks.
- `AYRSHARE_API_KEY` — `publish` attempts a live post via Ayrshare instead
  of writing a local outbox folder.

Nothing else changes. The output shapes (`Clip`, `Post`, `PublishResult`)
are identical either way.

## Stage-to-vendor mapping (if you want to swap in the original stack)

| Stage | v1 (zero-key) | Original paid-vendor idea | Swap-in note |
|---|---|---|---|
| Orchestration | local Typer CLI | Pipedream / Activepieces | every `pipeline/*.py` function is pure JSON-in/JSON-out — lift any stage into a Pipedream step directly |
| Ingest | `yt-dlp` + local file | Pipedream trigger/webhook | already free, no change needed |
| Transcribe | `faster-whisper` (CPU, no diarization) | WhisperX / OpenAI transcription API | add `pyannote` + a HuggingFace token to `transcribe.py` for diarization; or swap the whole function for an API call — same `Transcript` output shape |
| Clip selection | heuristic scorer | GPT-5.x / Claude via a clip-selector skill | set `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` — `clip_selector.py` already has the LLM path built in |
| Cut | `ffmpeg` | `ffmpeg` / Auto-Editor | as-is |
| Render/caption | `ffmpeg` burn-in | Remotion | `captioner.py`'s interface is narrow on purpose: raw clip + words in, finished mp4 out — write a Remotion-based module with that same signature and swap it in `cli.py`'s `caption` command, nothing else changes |
| Generate posts | templates | GPT / Claude via content-repurposer skill | same seam as clip selection |
| Publish | local outbox | Ayrshare / Postiz | set `AYRSHARE_API_KEY`; a Postiz adapter would live alongside `_publish_ayrshare` in `publisher.py` |
| Measure/learn | CSV + local stats | DB/Sheet + LLM | `analyst.py` is intentionally always-local/statistical; an LLM "why did these win" summary layer could sit on top of `Learnings` without changing its shape |

## Limitations of v1

- **No speaker diarization** — every transcript segment is tagged
  `SPEAKER_00`. Add `pyannote.audio` + a HuggingFace token to
  `transcribe.py` to change that.
- **ffmpeg burn-in captions only** — no branded Remotion templates yet (see
  mapping table above for the swap-in path).
- **No live publishing without a key** — `publish` always writes a local
  `outbox/` folder unless `AYRSHARE_API_KEY` is set; review before posting.
- **content-strategist has no offline fallback** — it's the one skill that's
  purely LLM-enhanced in v1 (see `skills/content-strategist/SKILL.md`).
  Without a key it's skipped; `clip_selector`'s heuristic runs standalone.
- **Heuristic clip scoring is simpler than an LLM's judgment** — it works
  well on transcripts with clear hook language and pauses, less well on
  monotone or jargon-heavy source audio. `analyst.py`'s `learnings.json`
  feedback loop is what's meant to close that gap over successive batches.
- **`yt-dlp` ingest is exercised in tests only at the CLI/argument level** —
  real YouTube downloads depend on external network conditions outside this
  repo's control; point `run` at a local file if you want a fully
  reproducible test run.

## Testing

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

70 tests, all passing with **zero external services and zero API keys**:

- `test_schemas.py`, `test_storage.py`, `test_llm_fallback.py` — pure Python,
  no binaries needed.
- `test_clip_selector.py`, `test_repurposer.py`, `test_analyst.py`,
  `test_publisher.py` — heuristic/template/statistical logic against fixture
  data in `fixtures/`.
- `test_media.py` — exercises real `ffmpeg` (cut + subtitle burn-in) against
  a synthetic video generated on the fly with `ffmpeg`'s `lavfi` source, so
  it needs no network and no sample media checked into the repo.
- `test_cli.py` — drives the full `run` command end-to-end (ingest → cut →
  caption → repurpose → publish) against that same synthetic video, with
  only the `transcribe` stage stubbed out (see below), and separately
  proves `--run-id` resume is idempotent.

### Manual end-to-end smoke test (with real transcription)

`transcribe.py`'s only external dependency is downloading the
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

This exact procedure was used to validate the pipeline while building it;
this build session's own network policy blocks `huggingface.co`, so the
model download itself couldn't be completed here — everything downstream of
transcription (`ffmpeg` cut/caption, template/heuristic logic, the full CLI
wiring and resume behavior) *was* verified against real binaries and a real
end-to-end `run`, with only that one model-download step stubbed in tests.
Run the command above on your own machine to confirm transcription too —
it should just work with normal internet access.

## Project layout

```
pipeline/            all stage modules + cli.py (see module docstrings)
skills/               the 4 SKILL.md docs (LLM-enabled contracts)
fixtures/              sample transcript + performance CSV used by tests
tests/                  pytest suite (70 tests, see Testing above)
runs/                    gitignored — per-run output (created on first use)
learnings.json           gitignored — written by `analyze`, read by
                          select-clips/repurpose on every subsequent run
```
