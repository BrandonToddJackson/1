"""Tests for pipeline.llm's JSON-parsing robustness: real LLMs routinely
wrap JSON in markdown fences, add leading/trailing prose, refuse, or hit a
token limit mid-response. None of that should crash the pipeline -- it
should raise a clear LLMResponseError that callers can catch and fall back
from. These tests never import the real openai/anthropic SDKs (matches
test_llm_fallback.py's style): client instances are built via
object.__new__ with fake SDK response objects attached by hand.
"""

import pytest

from pipeline import llm


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

def test_parses_plain_json():
    assert llm._parse_json_response('{"a": 1}') == {"a": 1}


def test_strips_markdown_fences():
    assert llm._parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm._parse_json_response('```\n{"a": 1}\n```') == {"a": 1}


def test_extracts_object_from_leading_and_trailing_prose():
    text = 'Here are the clips:\n{"clips": [1, 2, 3]}\nHope this helps!'
    assert llm._parse_json_response(text) == {"clips": [1, 2, 3]}


def test_ignores_braces_inside_strings():
    text = 'Sure! {"hook": "use {curly} braces wisely", "score": 1}'
    assert llm._parse_json_response(text) == {"hook": "use {curly} braces wisely", "score": 1}


def test_rejects_top_level_list():
    with pytest.raises(llm.LLMResponseError, match="expected a JSON object"):
        llm._parse_json_response("[1, 2, 3]")


def test_raises_llm_response_error_with_raw_text_on_garbage():
    with pytest.raises(llm.LLMResponseError) as exc_info:
        llm._parse_json_response("not json at all, no braces here")
    assert "not json at all" in str(exc_info.value)


def test_empty_response_raises():
    with pytest.raises(llm.LLMResponseError):
        llm._parse_json_response("")
    with pytest.raises(llm.LLMResponseError):
        llm._parse_json_response(None)


def test_whitespace_only_response_raises():
    with pytest.raises(llm.LLMResponseError):
        llm._parse_json_response("   \n  ")


# ---------------------------------------------------------------------------
# OpenAIClient.complete_json (fake SDK response, no real openai package)
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeOpenAIResponse:
    def __init__(self, choices):
        self.choices = choices


class _FakeOpenAISDKClient:
    """Stands in for `openai.OpenAI()` -- only implements the one call path
    OpenAIClient.complete_json actually uses."""

    def __init__(self, response):
        self._response = response
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return self._response


def _make_openai_client(response) -> "llm.OpenAIClient":
    client = object.__new__(llm.OpenAIClient)
    client._client = _FakeOpenAISDKClient(response)
    client._model = "fake-model"
    return client


def test_openai_valid_content_parses():
    client = _make_openai_client(_FakeOpenAIResponse([_FakeChoice('{"a": 1}')]))
    assert client.complete_json("sys", "user", "schema") == {"a": 1}


def test_openai_none_content_raises_llm_response_error():
    client = _make_openai_client(_FakeOpenAIResponse([_FakeChoice(None, finish_reason="length")]))
    with pytest.raises(llm.LLMResponseError, match="finish_reason=length"):
        client.complete_json("sys", "user", "schema")


def test_openai_no_choices_raises():
    client = _make_openai_client(_FakeOpenAIResponse([]))
    with pytest.raises(llm.LLMResponseError, match="no choices"):
        client.complete_json("sys", "user", "schema")


# ---------------------------------------------------------------------------
# AnthropicClient.complete_json (fake SDK response, no real anthropic package)
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeAnthropicSDKClient:
    def __init__(self, response):
        self._response = response
        self.messages = self

    def create(self, **kwargs):
        return self._response


def _make_anthropic_client(response) -> "llm.AnthropicClient":
    client = object.__new__(llm.AnthropicClient)
    client._client = _FakeAnthropicSDKClient(response)
    client._model = "fake-model"
    return client


def test_anthropic_valid_text_parses():
    client = _make_anthropic_client(_FakeAnthropicResponse([_FakeTextBlock('{"clips": []}')]))
    assert client.complete_json("sys", "user", "schema") == {"clips": []}


def test_anthropic_markdown_fenced_response_parses():
    client = _make_anthropic_client(_FakeAnthropicResponse([_FakeTextBlock('```json\n{"a": 1}\n```')]))
    assert client.complete_json("sys", "user", "schema") == {"a": 1}


def test_anthropic_empty_content_raises():
    client = _make_anthropic_client(_FakeAnthropicResponse([], stop_reason="max_tokens"))
    with pytest.raises(llm.LLMResponseError, match="stop_reason=max_tokens"):
        client.complete_json("sys", "user", "schema")
