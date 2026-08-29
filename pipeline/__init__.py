"""Zero-key content repurposing pipeline.

Long-form video/audio -> transcript -> best clips -> cut & captioned mp4s ->
platform-specific posts -> local outbox (or live publish) -> analytics ->
learnings fed back into the next run.

Every stage is a pure function that reads/writes JSON matching the pydantic
models in ``pipeline.schemas`` -- see README.md for the full architecture.
"""

__version__ = "0.1.0"
