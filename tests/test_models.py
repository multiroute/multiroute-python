from multiroute.providers import _load_providers, _extract_hostname, resolve_model

# --- already-prefixed model passthrough ---


def test_already_prefixed_is_unchanged():
    assert (
        resolve_model("openai/gpt-4o", "https://api.openai.com/v1/") == "openai/gpt-4o"
    )
    assert (
        resolve_model("anthropic/claude-3-opus", "https://api.anthropic.com")
        == "anthropic/claude-3-opus"
    )
    assert (
        resolve_model("groq/llama-3", "https://api.groq.com/openai/v1")
        == "groq/llama-3"
    )


# --- base_url-based provider resolution ---


def test_openai_base_url():
    assert resolve_model("gpt-4o", "https://api.openai.com/v1/") == "openai/gpt-4o"
    assert (
        resolve_model("gpt-4o-mini", "https://api.openai.com/v1/")
        == "openai/gpt-4o-mini"
    )
    assert resolve_model("o3-mini", "https://api.openai.com/v1/") == "openai/o3-mini"


def test_groq_base_url():
    assert resolve_model("gpt-4o", "https://api.groq.com/openai/v1") == "groq/gpt-4o"
    assert (
        resolve_model("llama-3-70b", "https://api.groq.com/openai/v1")
        == "groq/llama-3-70b"
    )


def test_anthropic_base_url():
    assert (
        resolve_model("claude-3-opus", "https://api.anthropic.com")
        == "anthropic/claude-3-opus"
    )
    assert (
        resolve_model("claude-3-5-sonnet", "https://api.anthropic.com/v1")
        == "anthropic/claude-3-5-sonnet"
    )


def test_google_base_url():
    assert (
        resolve_model("gemini-1.5-pro", "https://generativelanguage.googleapis.com/")
        == "google/gemini-1.5-pro"
    )


def test_mistral_base_url():
    assert (
        resolve_model("mistral-large", "https://api.mistral.ai/v1")
        == "mistral/mistral-large"
    )


def test_together_base_url():
    assert (
        resolve_model("llama-3-70b", "https://api.together.xyz/v1")
        == "together/llama-3-70b"
    )


def test_openrouter_base_url():
    assert (
        resolve_model("gpt-4o", "https://openrouter.ai/api/v1") == "openrouter/gpt-4o"
    )


def test_azure_base_url():
    assert (
        resolve_model("gpt-4o", "https://my-resource.openai.azure.com/")
        == "azure/gpt-4o"
    )


# --- no base_url or unknown URL → unchanged ---


def test_no_base_url_returns_model_unchanged():
    assert resolve_model("gpt-4o") == "gpt-4o"
    assert resolve_model("claude-3-opus") == "claude-3-opus"
    assert resolve_model("some-custom-model") == "some-custom-model"


def test_unknown_base_url_returns_model_unchanged():
    assert resolve_model("gpt-4o", "https://unknown-proxy.example.com") == "gpt-4o"
    assert resolve_model("my-model", "https://private-llm.internal") == "my-model"


def test_empty_model_is_unchanged():
    assert resolve_model("") == ""
    assert resolve_model("", "https://api.openai.com/v1/") == ""


# --- case insensitivity ---


def test_base_url_matching_is_case_insensitive():
    assert resolve_model("gpt-4o", "https://API.OPENAI.COM/v1/") == "openai/gpt-4o"
    assert resolve_model("gpt-4o", "https://Api.Groq.Com/openai/v1") == "groq/gpt-4o"


def test_model_casing_is_preserved():
    assert resolve_model("GPT-4O", "https://api.openai.com/v1/") == "openai/GPT-4O"
    assert (
        resolve_model("Claude-3-Opus", "https://api.anthropic.com")
        == "anthropic/Claude-3-Opus"
    )


# --- registry loading and hostname extraction ---


def test_providers_loads_once(monkeypatch):
    """_load_providers is cached; calling it twice should return the same object."""
    _load_providers.cache_clear()
    first = _load_providers()
    second = _load_providers()
    assert first is second


def test_providers_is_dict():
    """_load_providers returns a plain dict keyed by lowercase hostname."""
    providers = _load_providers()
    assert isinstance(providers, dict)
    assert providers.get("api.openai.com") == "openai"
    assert providers.get("api.groq.com") == "groq"


def test_extract_hostname():
    assert _extract_hostname("https://api.openai.com/v1/") == "api.openai.com"
    assert _extract_hostname("https://api.groq.com/openai/v1") == "api.groq.com"
    assert _extract_hostname("https://API.OPENAI.COM/v1/") == "api.openai.com"
    assert _extract_hostname("not-a-url") is None
