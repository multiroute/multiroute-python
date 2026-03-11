import pytest
import respx
import httpx
from openai import APIConnectionError, InternalServerError, APITimeoutError
from multiroute.openai import OpenAI, AsyncOpenAI

@pytest.fixture
def client():
    return OpenAI(api_key="test-key", max_retries=0)

@pytest.fixture
def async_client():
    return AsyncOpenAI(api_key="test-key", max_retries=0)

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "fake")

@respx.mock
def test_chat_completions_success(client):
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        })
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "Hello there!"
    assert multiroute_route.called
    assert not openai_route.called
    
    # Check the request that was made to multiroute
    request = multiroute_route.calls.last.request
    assert request.headers["Authorization"] == "Bearer fake"

@respx.mock
def test_chat_completions_fallback_500(client):
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "Internal Server Error"}})
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Fallback response!"
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        })
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "Fallback response!"
    assert multiroute_route.called
    assert openai_route.called


@respx.mock
def test_chat_completions_fallback_connection_error(client):
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Fallback response after connection error!"
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        })
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "Fallback response after connection error!"
    assert multiroute_route.called
    assert openai_route.called


@respx.mock
async def test_async_chat_completions_fallback_500(async_client):
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "Internal Server Error"}})
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0125",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Async Fallback response!"
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        })
    )
    
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "Async Fallback response!"
    assert multiroute_route.called
    assert openai_route.called

@respx.mock
def test_chat_completions_fallback_404(client):
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"detail": "Not Found"})
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "404 Fallback!"}}],
            "usage": {"total_tokens": 10}
        })
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "404 Fallback!"
    assert multiroute_route.called
    assert openai_route.called


@respx.mock
def test_chat_completions_no_multiroute_key(client, monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    
    openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-3.5-turbo",
            "choices": [{"message": {"role": "assistant", "content": "Direct OpenAI!"}}],
            "usage": {"total_tokens": 10}
        })
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    assert response.choices[0].message.content == "Direct OpenAI!"
    assert not multiroute_route.called
    assert openai_route.called

