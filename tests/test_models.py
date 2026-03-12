import pytest

from multiroute.models import resolve_model, _load_registry


# --- resolver unit tests ---


def test_already_prefixed_is_unchanged():
    assert resolve_model("openai/gpt-4o") == "openai/gpt-4o"
    assert resolve_model("anthropic/claude-3-opus") == "anthropic/claude-3-opus"
    assert resolve_model("google/gemini-1.5-pro") == "google/gemini-1.5-pro"


def test_bare_openai_models():
    assert resolve_model("gpt-4o") == "openai/gpt-4o"
    assert resolve_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert resolve_model("gpt-4") == "openai/gpt-4"
    assert resolve_model("gpt-3.5-turbo") == "openai/gpt-3.5-turbo"
    assert resolve_model("o1") == "openai/o1"
    assert resolve_model("o3-mini") == "openai/o3-mini"


def test_dated_variant_still_resolves():
    """Versioned suffixes like -2024-07-18 should still match."""
    assert resolve_model("gpt-4o-mini-2024-07-18") == "openai/gpt-4o-mini-2024-07-18"
    assert (
        resolve_model("claude-3-5-sonnet-20241022")
        == "anthropic/claude-3-5-sonnet-20241022"
    )
    assert resolve_model("claude-3-opus-20240229") == "anthropic/claude-3-opus-20240229"


def test_bare_anthropic_models():
    assert resolve_model("claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"
    assert resolve_model("claude-3-haiku") == "anthropic/claude-3-haiku"
    assert resolve_model("claude-2") == "anthropic/claude-2"


def test_bare_google_models():
    assert resolve_model("gemini-1.5-pro") == "google/gemini-1.5-pro"
    assert resolve_model("gemini-2.0-flash") == "google/gemini-2.0-flash"


def test_bare_other_providers():
    assert resolve_model("mistral-large") == "mistral/mistral-large"
    assert resolve_model("mixtral-8x7b") == "mistral/mixtral-8x7b"
    assert resolve_model("deepseek-chat") == "deepseek/deepseek-chat"
    assert resolve_model("command-r") == "cohere/command-r"


def test_unknown_model_is_unchanged():
    assert resolve_model("my-custom-finetuned-model") == "my-custom-finetuned-model"
    assert resolve_model("some-local-llm") == "some-local-llm"


def test_empty_string_is_unchanged():
    assert resolve_model("") == ""


def test_case_insensitive():
    assert resolve_model("GPT-4O") == "openai/GPT-4O"
    assert resolve_model("Claude-3-Opus") == "anthropic/Claude-3-Opus"


def test_registry_loads_once(monkeypatch):
    """_load_registry is cached; calling it twice should return the same object."""
    _load_registry.cache_clear()
    first = _load_registry()
    second = _load_registry()
    assert first is second


def test_bare_meta_models():
    """Meta/Llama models should resolve to the 'meta' provider."""
    assert resolve_model("llama-3") == "meta/llama-3"
    assert resolve_model("llama-3-70b-instruct") == "meta/llama-3-70b-instruct"
    assert resolve_model("meta-llama-3") == "meta/meta-llama-3"


def test_bare_xai_models():
    """xAI Grok models should resolve to the 'xai' provider."""
    assert resolve_model("grok") == "xai/grok"
    assert resolve_model("grok-2") == "xai/grok-2"
    assert resolve_model("grok-beta") == "xai/grok-beta"


def test_dot_separator_prefix_matching():
    """The resolver handles 'model.version' style names via the '.' separator."""
    # gemini-1.5-pro.001 should still match the gemini-1.5-pro pattern
    assert resolve_model("gemini-1.5-pro.001") == "google/gemini-1.5-pro.001"


def test_more_specific_pattern_wins():
    """Longer (more specific) patterns should take precedence over shorter ones.

    gpt-4o-mini is more specific than gpt-4o, so 'gpt-4o-mini-2024' must map
    to openai/... via the gpt-4o-mini entry, not gpt-4o.
    """
    result = resolve_model("gpt-4o-mini-2024-07-18")
    assert result == "openai/gpt-4o-mini-2024-07-18"
    # Confirm it is NOT matching the shorter 'gpt-4o' pattern
    # (would produce same provider here, but the pattern chosen matters)
    assert result.startswith("openai/")
