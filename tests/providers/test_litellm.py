from unittest.mock import AsyncMock, patch

import pytest
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    NotFoundError,
    Timeout,
)

from multiroute.litellm import acompletion, completion


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "test-multiroute-key")


def test_litellm_completion_proxy_success(mock_env):
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "proxy_success"

        response = completion(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "proxy_success"
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "claude-3-opus"
        assert kwargs["api_base"] == "https://api.multiroute.ai/openai/v1"
        assert kwargs["api_key"] == "test-multiroute-key"
        assert kwargs["custom_llm_provider"] == "openai"


def test_litellm_completion_proxy_fallback(mock_env):
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        # First call fails with 500, second call succeeds
        error = InternalServerError(
            message="Proxy error",
            model="claude-3-opus",
            llm_provider="openai",
        )
        mock_completion.side_effect = [error, "fallback_success"]

        response = completion(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "fallback_success"
        assert mock_completion.call_count == 2

        # Check first call (proxy)
        proxy_kwargs = mock_completion.call_args_list[0].kwargs
        assert proxy_kwargs["api_base"] == "https://api.multiroute.ai/openai/v1"
        assert proxy_kwargs["custom_llm_provider"] == "openai"

        # Check second call (fallback)
        fallback_kwargs = mock_completion.call_args_list[1].kwargs
        assert fallback_kwargs["model"] == "claude-3-opus"
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


@pytest.mark.asyncio
async def test_litellm_acompletion_proxy_fallback(mock_env):
    with patch(
        "multiroute.litellm.client.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_acompletion:
        error = InternalServerError(
            message="Proxy error",
            model="gpt-4",
            llm_provider="openai",
        )
        mock_acompletion.side_effect = [error, "async_fallback_success"]

        response = await acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "async_fallback_success"
        assert mock_acompletion.call_count == 2

        proxy_kwargs = mock_acompletion.call_args_list[0].kwargs
        assert proxy_kwargs["api_base"] == "https://api.multiroute.ai/openai/v1"
        assert proxy_kwargs["custom_llm_provider"] == "openai"

        fallback_kwargs = mock_acompletion.call_args_list[1].kwargs
        assert fallback_kwargs["model"] == "gpt-4"
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


def test_litellm_completion_no_key(monkeypatch, caplog):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    import logging

    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "direct_success"

        with caplog.at_level(logging.ERROR):
            response = completion(
                model="claude-3-opus",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert "MULTIROUTE_API_KEY is not set" in caplog.text
        assert response == "direct_success"
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        assert "api_base" not in kwargs
        assert "custom_llm_provider" not in kwargs


# ---------------------------------------------------------------------------
# acompletion — no key path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_acompletion_no_key(monkeypatch, caplog):
    """Acompletion without MULTIROUTE_API_KEY calls litellm directly."""
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    import logging

    with patch(
        "multiroute.litellm.client.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_acompletion:
        mock_acompletion.return_value = "direct_async_success"

        with caplog.at_level(logging.ERROR):
            response = await acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert "MULTIROUTE_API_KEY is not set" in caplog.text
        assert response == "direct_async_success"
        mock_acompletion.assert_called_once()
        kwargs = mock_acompletion.call_args.kwargs
        assert "api_base" not in kwargs
        assert "custom_llm_provider" not in kwargs


# ---------------------------------------------------------------------------
# Connection error triggers fallback
# ---------------------------------------------------------------------------


def test_litellm_completion_connection_error_fallback(mock_env):
    """An APIConnectionError from the proxy triggers fallback to native litellm."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        error = APIConnectionError(
            message="Connection refused",
            model="gpt-4o",
            llm_provider="openai",
        )
        mock_completion.side_effect = [error, "conn_fallback_success"]

        response = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "conn_fallback_success"
        assert mock_completion.call_count == 2

        fallback_kwargs = mock_completion.call_args_list[1].kwargs
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


# ---------------------------------------------------------------------------
# Non-multiroute errors are re-raised
# ---------------------------------------------------------------------------


def test_litellm_completion_non_multiroute_error_reraised(mock_env):
    """A ValueError (non-multiroute error) from the proxy is re-raised, not swallowed."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            completion(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])

        # Only one call — no fallback attempted
        assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# resolve_model prefix applied to proxy call
# ---------------------------------------------------------------------------


def test_litellm_completion_resolve_model_prefixes_gpt(mock_env):
    """Model names are passed through to the proxy unchanged (no prefix added for litellm)."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "ok"

        completion(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["api_base"] == "https://api.multiroute.ai/openai/v1"


def test_litellm_completion_unknown_model_unchanged(mock_env):
    """An unknown model name should be passed through unchanged to the proxy."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "ok"

        completion(
            model="my-custom-local-llm",
            messages=[{"role": "user", "content": "hi"}],
        )

        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        # Unknown model stays unprefixed
        assert kwargs["model"] == "my-custom-local-llm"


# ---------------------------------------------------------------------------
# 404 fallback
# ---------------------------------------------------------------------------


def test_litellm_completion_404_fallback(mock_env):
    """A NotFoundError (404) from the proxy triggers fallback to native litellm."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        error = NotFoundError(
            message="Not Found",
            model="gpt-4o",
            llm_provider="openai",
        )
        mock_completion.side_effect = [error, "404_fallback_success"]

        response = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "404_fallback_success"
        assert mock_completion.call_count == 2

        fallback_kwargs = mock_completion.call_args_list[1].kwargs
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


# ---------------------------------------------------------------------------
# Timeout fallback
# ---------------------------------------------------------------------------


def test_litellm_completion_timeout_fallback(mock_env):
    """A Timeout error from the proxy triggers fallback to native litellm."""
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        error = Timeout(
            message="Request timed out",
            model="gpt-4o",
            llm_provider="openai",
        )
        mock_completion.side_effect = [error, "timeout_fallback_success"]

        response = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert response == "timeout_fallback_success"
        assert mock_completion.call_count == 2

        fallback_kwargs = mock_completion.call_args_list[1].kwargs
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs
