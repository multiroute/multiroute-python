import json
import logging

import httpx
import pytest
import respx

from multiroute.openai.client import MULTIROUTE_BASE_URL
from multiroute.pydantic_ai import Agent, MultirouteOpenAIProvider

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_OPENAI_SUCCESS_JSON = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from multiroute!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
}

_OPENAI_NATIVE_JSON = {
    "id": "chatcmpl-native",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from native OpenAI!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
}


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "fake-mr-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")


def _make_agent():
    """Return a (Agent, OpenAIChatModel) pair using MultirouteOpenAIProvider."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel

    model = OpenAIChatModel("gpt-4o", provider=MultirouteOpenAIProvider())
    agent = Agent(model)
    return agent, model


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@respx.mock
async def test_success_routes_through_proxy():
    """Happy path: proxy is called, native OpenAI is not, response is correct."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Say hello")

    assert result.output == "Hello from multiroute!"
    assert proxy_route.called
    assert not native_route.called


@respx.mock
async def test_proxy_receives_multiroute_api_key():
    """The proxy request must carry the MULTIROUTE_API_KEY as Bearer token."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )

    agent, _ = _make_agent()
    await agent.run("Hi")

    assert proxy_route.called
    auth = proxy_route.calls.last.request.headers.get("Authorization", "")
    assert auth == "Bearer fake-mr-key"


@respx.mock
async def test_model_name_prefixed_for_proxy():
    """Model name must be provider-prefixed (openai/gpt-4o) in the proxy request."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )

    agent, _ = _make_agent()
    await agent.run("Hi")

    req_body = json.loads(proxy_route.calls.last.request.content)
    assert req_body["model"] == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# Fallback on proxy errors
# ---------------------------------------------------------------------------


@respx.mock
async def test_fallback_on_proxy_500():
    """5xx from the proxy triggers fallback to native OpenAI."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Hi")

    assert proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


@respx.mock
async def test_fallback_on_proxy_connection_error():
    """Connection error from the proxy triggers fallback to native OpenAI."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Hi")

    assert proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


@respx.mock
async def test_fallback_on_proxy_404():
    """404 from the proxy triggers fallback to native OpenAI."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, json={"detail": "Not Found"})
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Hi")

    assert proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


@respx.mock
async def test_fallback_on_proxy_timeout():
    """Timeout from the proxy triggers fallback to native OpenAI."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Hi")

    assert proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


# ---------------------------------------------------------------------------
# No MULTIROUTE_API_KEY — skip proxy entirely
# ---------------------------------------------------------------------------


@respx.mock
async def test_no_multiroute_key_skips_proxy(monkeypatch):
    """When MULTIROUTE_API_KEY is absent, requests go directly to native OpenAI."""
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent, _ = _make_agent()
    result = await agent.run("Hi")

    assert not proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


# ---------------------------------------------------------------------------
# Non-multiroute errors are re-raised
# ---------------------------------------------------------------------------


@respx.mock
async def test_non_multiroute_error_reraised():
    """A 401 auth error from the proxy must be re-raised, not swallowed."""
    from pydantic_ai import ModelHTTPError

    respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            },
        )
    )

    agent, _ = _make_agent()
    with pytest.raises(ModelHTTPError):
        await agent.run("Hi")


# ---------------------------------------------------------------------------
# Warning logged when MULTIROUTE_API_KEY is missing
# ---------------------------------------------------------------------------


def test_no_key_logs_error(monkeypatch, caplog):
    """Constructing the provider without MULTIROUTE_API_KEY logs an error."""
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    with caplog.at_level(logging.ERROR):
        MultirouteOpenAIProvider()

    assert "MULTIROUTE_API_KEY is not set" in caplog.text


# ---------------------------------------------------------------------------
# Agent with string model names (no make_model() required)
# ---------------------------------------------------------------------------


@respx.mock
async def test_agent_string_model_routes_through_proxy():
    """Agent('openai:gpt-4o') automatically routes through the Multiroute proxy."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent = Agent("openai:gpt-4o")
    result = await agent.run("Say hello")

    assert result.output == "Hello from multiroute!"
    assert proxy_route.called
    assert not native_route.called


@respx.mock
async def test_agent_prefixed_string_model_routes_through_proxy():
    """Agent('openai:gpt-4o') (provider-prefixed) also routes through the proxy."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent = Agent("openai:gpt-4o")
    result = await agent.run("Say hello")

    assert result.output == "Hello from multiroute!"
    assert proxy_route.called
    assert not native_route.called


@respx.mock
async def test_agent_string_model_fallback_on_proxy_500():
    """Agent('openai:gpt-4o') falls back to native OpenAI when the proxy returns 5xx."""
    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent = Agent("openai:gpt-4o")
    result = await agent.run("Hi")

    assert proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"


@respx.mock
async def test_agent_string_model_no_multiroute_key_skips_proxy(monkeypatch):
    """Agent('openai:gpt-4o') goes directly to native OpenAI when MULTIROUTE_API_KEY is absent."""
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    proxy_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_SUCCESS_JSON)
    )
    native_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_NATIVE_JSON)
    )

    agent = Agent("openai:gpt-4o")
    result = await agent.run("Hi")

    assert not proxy_route.called
    assert native_route.called
    assert result.output == "Hello from native OpenAI!"
