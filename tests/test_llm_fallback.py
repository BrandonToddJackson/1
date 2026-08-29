"""Tests for the zero-key seam: get_llm_client() must return None with no
keys configured, and must not require the openai/anthropic SDKs to be
installed to prove that routing logic works.
"""

import sys

import pytest

from pipeline import llm


def _clear_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_no_keys_returns_none(monkeypatch):
    _clear_keys(monkeypatch)
    assert llm.get_llm_client() is None


def test_anthropic_key_selected_when_only_anthropic_set(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    class FakeAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AnthropicClient", FakeAnthropicClient)

    client = llm.get_llm_client()
    assert isinstance(client, FakeAnthropicClient)
    assert client.api_key == "fake-key"


def test_openai_key_selected_when_only_openai_set(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    class FakeOpenAIClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "OpenAIClient", FakeOpenAIClient)

    client = llm.get_llm_client()
    assert isinstance(client, FakeOpenAIClient)


def test_anthropic_preferred_when_both_keys_set(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    class FakeAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    class FakeOpenAIClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AnthropicClient", FakeAnthropicClient)
    monkeypatch.setattr(llm, "OpenAIClient", FakeOpenAIClient)

    client = llm.get_llm_client()
    assert isinstance(client, FakeAnthropicClient)


def test_real_client_classes_import_lazily(monkeypatch):
    """Instantiating without the optional SDK available must fail with an
    ImportError (from the lazy import), not at module import time -- proving
    `import pipeline.llm` itself never requires openai/anthropic installed.

    Forces the import to fail via sys.modules rather than relying on openai
    actually being absent from this venv -- the previous version of this
    test passed only because the package happened not to be installed here,
    and silently started failing the moment it was (e.g. to verify a real
    key), which tested this environment's install state, not the contract.
    """
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setitem(sys.modules, "openai", None)  # sys.modules[x] = None forces ImportError on `import x`
    with pytest.raises(ImportError):
        llm.get_llm_client()
