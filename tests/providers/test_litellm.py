import pytest
from unittest.mock import patch, AsyncMock
from litellm.exceptions import InternalServerError

from multiroute.litellm import completion, acompletion


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "test-multiroute-key")


def test_litellm_completion_proxy_success(mock_env):
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "proxy_success"

        response = completion(
            model="claude-3-opus", messages=[{"role": "user", "content": "hello"}]
        )

        assert response == "proxy_success"
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "claude-3-opus"
        assert kwargs["api_base"] == "https://api.multiroute.ai/v1"
        assert kwargs["api_key"] == "test-multiroute-key"
        assert kwargs["custom_llm_provider"] == "openai"


def test_litellm_completion_proxy_fallback(mock_env):
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        # First call fails with 500, second call succeeds
        error = InternalServerError(
            message="Proxy error", model="claude-3-opus", llm_provider="openai"
        )
        mock_completion.side_effect = [error, "fallback_success"]

        response = completion(
            model="claude-3-opus", messages=[{"role": "user", "content": "hello"}]
        )

        assert response == "fallback_success"
        assert mock_completion.call_count == 2

        # Check first call (proxy)
        proxy_kwargs = mock_completion.call_args_list[0].kwargs
        assert proxy_kwargs["api_base"] == "https://api.multiroute.ai/v1"
        assert proxy_kwargs["custom_llm_provider"] == "openai"

        # Check second call (fallback)
        fallback_kwargs = mock_completion.call_args_list[1].kwargs
        assert fallback_kwargs["model"] == "claude-3-opus"
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


@pytest.mark.asyncio
async def test_litellm_acompletion_proxy_fallback(mock_env):
    with patch(
        "multiroute.litellm.client.litellm.acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        error = InternalServerError(
            message="Proxy error", model="gpt-4", llm_provider="openai"
        )
        mock_acompletion.side_effect = [error, "async_fallback_success"]

        response = await acompletion(
            model="gpt-4", messages=[{"role": "user", "content": "hello"}]
        )

        assert response == "async_fallback_success"
        assert mock_acompletion.call_count == 2

        proxy_kwargs = mock_acompletion.call_args_list[0].kwargs
        assert proxy_kwargs["api_base"] == "https://api.multiroute.ai/v1"
        assert proxy_kwargs["custom_llm_provider"] == "openai"

        fallback_kwargs = mock_acompletion.call_args_list[1].kwargs
        assert fallback_kwargs["model"] == "gpt-4"
        assert "api_base" not in fallback_kwargs
        assert "custom_llm_provider" not in fallback_kwargs


def test_litellm_completion_no_key(monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    with patch("multiroute.litellm.client.litellm.completion") as mock_completion:
        mock_completion.return_value = "direct_success"

        response = completion(
            model="claude-3-opus", messages=[{"role": "user", "content": "hello"}]
        )

        assert response == "direct_success"
        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        assert "api_base" not in kwargs
        assert "custom_llm_provider" not in kwargs
