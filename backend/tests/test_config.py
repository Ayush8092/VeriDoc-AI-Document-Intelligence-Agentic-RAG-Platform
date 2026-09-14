from app.core.config import Settings

# Settings(_env_file=None) only disables reading .env — pydantic-settings
# still reads real process environment variables on top of it. CI sets
# GEMINI_API_KEY/GROQ_API_KEY/PINECONE_API_KEY/DATABASE_URL/JWT_SECRET_KEY
# at the job level (for the S3/Postgres integration tests), so these
# "defaults" tests must explicitly clear them first or they'll pick up
# whatever the runner happens to have set instead of the class defaults.
_AMBIENT_ENV_VARS = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "DATABASE_URL",
    "JWT_SECRET_KEY",
)


def _clear_ambient_env(monkeypatch):
    for var in _AMBIENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_load_with_no_api_keys_set(monkeypatch):
    """API keys are optional at import/construction time — they only fail
    loudly when a client that needs them is actually built (see
    app/clients.py). This is what lets the app, its docs, and most of its
    test suite run without every credential configured.
    """
    _clear_ambient_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key is None
    assert settings.groq_api_key is None
    assert settings.pinecone_api_key is None


def test_settings_defaults(monkeypatch):
    _clear_ambient_env(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.app_name == "Veridoc"
    assert settings.top_k == 10
    assert settings.rerank_enabled is True
    assert settings.database_url.startswith("sqlite")
    assert settings.ocr_language == "eng"


def test_settings_reads_env_override(monkeypatch):
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("OCR_LANGUAGE", "fra")
    settings = Settings(_env_file=None)
    assert settings.top_k == 5
    assert settings.ocr_language == "fra"