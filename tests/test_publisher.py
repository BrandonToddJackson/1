import json

import pytest

from pipeline import publisher
from pipeline.schemas import Post


@pytest.fixture
def posts() -> list[Post]:
    return [
        Post(platform="x", clip_id="clip-01", text="check this out", hashtags=["#money"], cta=None),
        Post(platform="linkedin", clip_id="clip-01", text="a longer take", hashtags=["#money", "#saving"], cta="What do you think?"),
    ]


def test_publish_with_no_key_writes_local_outbox(monkeypatch, tmp_path, posts):
    monkeypatch.delenv("AYRSHARE_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    results = publisher.publish("run-1", posts)

    assert len(results) == 2
    assert all(r.method == "outbox" for r in results)
    assert all(r.status == "ready" for r in results)

    x_dir = tmp_path / "run-1" / "outbox" / "clip-01" / "x"
    assert (x_dir / "caption.txt").exists()
    assert (x_dir / "metadata.json").exists()

    caption = (x_dir / "caption.txt").read_text()
    assert "check this out" in caption
    assert "#money" in caption

    metadata = json.loads((x_dir / "metadata.json").read_text())
    assert metadata["platform"] == "x"


def test_publish_copies_media_when_provided(monkeypatch, tmp_path, posts):
    monkeypatch.delenv("AYRSHARE_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    fake_media = tmp_path / "clip-01.mp4"
    fake_media.write_bytes(b"not really a video")

    publisher.publish("run-1", posts, clip_media={"clip-01": fake_media})

    x_media = tmp_path / "run-1" / "outbox" / "clip-01" / "x" / "media.mp4"
    assert x_media.exists()
    assert x_media.read_bytes() == b"not really a video"


def test_publish_uses_ayrshare_when_key_present(monkeypatch, tmp_path, posts):
    monkeypatch.setenv("AYRSHARE_API_KEY", "fake-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(publisher, "_publish_ayrshare", lambda key, run_id, posts, media: "called-ayrshare")

    result = publisher.publish("run-1", posts)
    assert result == "called-ayrshare"
