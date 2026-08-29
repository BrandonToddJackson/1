import json

import pytest

from pipeline import publisher
from pipeline.schemas import Post, PublishResult


@pytest.fixture
def posts() -> list[Post]:
    return [
        Post(platform="x", clip_id="clip-01", text="check this out", hashtags=["#money"], cta=None),
        Post(platform="linkedin", clip_id="clip-01", text="a longer take", hashtags=["#money", "#saving"], cta="What do you think?"),
    ]


def _clear_keys(monkeypatch):
    monkeypatch.delenv("AYRSHARE_API_KEY", raising=False)
    monkeypatch.delenv("BLOTATO_API_KEY", raising=False)
    monkeypatch.delenv("BLOTATO_ACCOUNT_IDS", raising=False)


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class _FakeRequestsModule:
    """Stands in for the `requests` module so tests never need it installed
    or make a real network call -- captures every .post() call for
    inspection."""

    def __init__(self, response_json=None):
        self._response_json = response_json if response_json is not None else {"id": "fake-id"}
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(self._response_json)


def test_publish_with_no_key_writes_local_outbox(monkeypatch, tmp_path, posts):
    _clear_keys(monkeypatch)
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
    _clear_keys(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    fake_media = tmp_path / "clip-01.mp4"
    fake_media.write_bytes(b"not really a video")

    publisher.publish("run-1", posts, clip_media={"clip-01": fake_media})

    x_media = tmp_path / "run-1" / "outbox" / "clip-01" / "x" / "media.mp4"
    assert x_media.exists()
    assert x_media.read_bytes() == b"not really a video"


def test_publish_uses_ayrshare_when_key_present(monkeypatch, tmp_path, posts):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("AYRSHARE_API_KEY", "fake-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    captured = {}

    def fake_publish_ayrshare(api_key, live_posts):
        captured["api_key"] = api_key
        return [PublishResult(platform=p.platform, method="ayrshare", location="fake-id", status="published") for p in live_posts]

    monkeypatch.setattr(publisher, "_publish_ayrshare", fake_publish_ayrshare)

    results = publisher.publish("run-1", posts)
    assert captured["api_key"] == "fake-key"
    assert len(results) == 2
    assert all(r.method == "ayrshare" and r.status == "published" for r in results)


def test_blotato_preferred_over_ayrshare_when_both_keys_set(monkeypatch, tmp_path, posts):
    monkeypatch.setenv("BLOTATO_API_KEY", "blotato-key")
    monkeypatch.setenv("AYRSHARE_API_KEY", "ayrshare-key")
    monkeypatch.setenv("BLOTATO_ACCOUNT_IDS", json.dumps({"x": "acct-x", "linkedin": "acct-li"}))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    results = publisher.publish("run-1", posts)
    assert all(r.method == "blotato" for r in results)
    assert all(call["url"] == "https://backend.blotato.com/v2/posts" for call in fake.calls)


def test_outbox_caption_respects_hard_limit_and_keeps_hashtags(monkeypatch, tmp_path):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    long_text = "x" * 300  # exceeds X's 280-char hard limit on its own
    post = Post(platform="x", clip_id="clip-01", text=long_text, hashtags=["#money", "#saving"], cta=None)

    publisher.publish("run-1", [post])

    caption = (tmp_path / "run-1" / "outbox" / "clip-01" / "x" / "caption.txt").read_text().rstrip("\n")
    assert len(caption) <= 280
    assert "#money" in caption
    assert "#saving" in caption


def test_ayrshare_payload_includes_cta_and_hashtags(monkeypatch, posts):
    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    publisher._publish_ayrshare("fake-key", posts)

    linkedin_call = next(c for c in fake.calls if c["json"]["platforms"] == ["linkedin"])
    assert "What do you think?" in linkedin_call["json"]["post"]
    assert "#money" in linkedin_call["json"]["post"]


def test_ayrshare_maps_x_to_twitter_and_shorts_to_youtube(monkeypatch):
    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    posts = [Post(platform="x", text="a"), Post(platform="shorts", text="b")]
    publisher._publish_ayrshare("fake-key", posts)

    platforms_sent = [c["json"]["platforms"][0] for c in fake.calls]
    assert platforms_sent == ["twitter", "youtube"]


def test_newsletter_routes_to_outbox_when_live_key_set(monkeypatch, tmp_path):
    monkeypatch.setenv("AYRSHARE_API_KEY", "fake-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    post = Post(platform="newsletter", clip_id="clip-01", text="Subject: hi\n\nBody", hashtags=[], cta=None)
    results = publisher.publish("run-1", [post])

    assert results[0].method == "outbox"
    assert not fake.calls  # never called the live API for a platform it doesn't support
    assert (tmp_path / "run-1" / "outbox" / "clip-01" / "newsletter" / "caption.txt").exists()


def test_missing_requests_reports_failed_per_post_not_crash(monkeypatch, tmp_path, posts):
    monkeypatch.setenv("AYRSHARE_API_KEY", "fake-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(publisher, "_get_requests", lambda: None)

    results = publisher.publish("run-1", posts)  # must not raise
    assert len(results) == 2
    assert all(r.status == "failed" for r in results)
    assert all("requests not installed" in r.location for r in results)


def test_blotato_payload_shape(monkeypatch):
    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    post = Post(platform="x", text="hello world", hashtags=[], cta=None)
    results = publisher._publish_blotato("fake-key", {"x": "acct-123"}, [post])

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == "https://backend.blotato.com/v2/posts"
    assert call["headers"]["blotato-api-key"] == "fake-key"
    assert call["json"]["post"]["accountId"] == "acct-123"
    assert call["json"]["post"]["content"]["platform"] == "twitter"
    assert call["json"]["post"]["target"]["targetType"] == "twitter"
    assert "hello world" in call["json"]["post"]["content"]["text"]
    assert results[0].status == "published"


def test_blotato_missing_account_id_reports_failure(monkeypatch):
    fake = _FakeRequestsModule()
    monkeypatch.setattr(publisher, "_get_requests", lambda: fake)

    post = Post(platform="x", text="hello", hashtags=[], cta=None)
    results = publisher._publish_blotato("fake-key", {}, [post])  # no account id configured

    assert results[0].status == "failed"
    assert "BLOTATO_ACCOUNT_IDS" in results[0].location
    assert not fake.calls  # never attempted the request


def test_build_caption_includes_cta_and_hashtags():
    post = Post(platform="newsletter", text="Subject: x\n\nBody", hashtags=[], cta="Reply!")
    caption = publisher.build_caption(post)
    assert "Reply!" in caption


def test_build_caption_preserves_hashtags_when_shrinking_for_hard_limit():
    post = Post(platform="x", text="y" * 300, hashtags=["#a", "#b"], cta=None)
    caption = publisher.build_caption(post)
    assert len(caption) <= 280
    assert "#a" in caption and "#b" in caption
