from pipeline import config


def _clear_all(monkeypatch):
    for var in (
        "DATA_DIR", "WHISPER_MODEL", "WHISPER_DEVICE", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", "AYRSHARE_API_KEY", "BLOTATO_API_KEY", "BLOTATO_ACCOUNT_IDS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_get_settings_defaults(monkeypatch):
    _clear_all(monkeypatch)
    settings = config.get_settings()
    assert settings.whisper_model == "base"
    assert settings.blotato_api_key is None
    assert settings.blotato_account_ids == {}


def test_blotato_account_ids_parsed_from_json(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("BLOTATO_ACCOUNT_IDS", '{"x": "acct_1", "linkedin": "acct_2"}')
    settings = config.get_settings()
    assert settings.blotato_account_ids == {"x": "acct_1", "linkedin": "acct_2"}


def test_blotato_account_ids_malformed_json_ignored(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("BLOTATO_ACCOUNT_IDS", "not json")
    settings = config.get_settings()
    assert settings.blotato_account_ids == {}


def test_blotato_account_ids_non_object_json_ignored(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("BLOTATO_ACCOUNT_IDS", "[1, 2, 3]")
    settings = config.get_settings()
    assert settings.blotato_account_ids == {}


def test_blotato_api_key_read_from_env(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("BLOTATO_API_KEY", "fake-key")
    settings = config.get_settings()
    assert settings.blotato_api_key == "fake-key"
