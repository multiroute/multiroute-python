import pytest
import respx
import httpx
from anthropic import APIConnectionError, InternalServerError, APITimeoutError
from multiroute.anthropic import Anthropic, AsyncAnthropic
from multiroute.anthropic.client import MULTIROUTE_BASE_URL

@pytest.fixture
def client():
    return Anthropic(api_key="test-key", max_retries=0)

@pytest.fixture
def async_client():
    return AsyncAnthropic(api_key="test-key", max_retries=0)

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "fake")

@respx.mock
def test_messages_success(client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
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
    
    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={})
    )
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
    
    assert response.content[0].text == "Hello there!"
    assert response.role == "assistant"
    assert response.id == "chatcmpl-123"
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 12
    
    assert multiroute_route.called
    assert not anthropic_route.called
    
    # Check the request that was made to multiroute
    request = multiroute_route.calls.last.request
    assert request.headers["Authorization"] == "Bearer fake"
    
    req_json = import_json(request.content)
    assert req_json["model"] == "claude-3-opus-20240229"
    assert req_json["messages"] == [{"role": "user", "content": "Hello!"}]
    assert req_json["max_tokens"] == 100

def import_json(content):
    import json
    return json.loads(content)


@respx.mock
def test_messages_fallback_500(client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "Internal Server Error"}})
    )
    
    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {
                    "type": "text",
                    "text": "Fallback response!"
                }
            ],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12
            }
        })
    )
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
    
    assert response.content[0].text == "Fallback response!"
    assert multiroute_route.called
    assert anthropic_route.called


@respx.mock
def test_messages_fallback_connection_error(client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    
    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {
                    "type": "text",
                    "text": "Fallback response after connection error!"
                }
            ],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12
            }
        })
    )
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
    
    assert response.content[0].text == "Fallback response after connection error!"
    assert multiroute_route.called
    assert anthropic_route.called

@respx.mock
async def test_async_messages_fallback_500(async_client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "Internal Server Error"}})
    )
    
    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {
                    "type": "text",
                    "text": "Async Fallback response!"
                }
            ],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12
            }
        })
    )
    
    response = await async_client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
    
    assert response.content[0].text == "Async Fallback response!"
    assert multiroute_route.called
    assert anthropic_route.called


@respx.mock
def test_messages_no_multiroute_key(client, monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    
    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [
                {
                    "type": "text",
                    "text": "Direct Anthropic!"
                }
            ],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 12
            }
        })
    )
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100
    )
    
    assert response.content[0].text == "Direct Anthropic!"
    assert not multiroute_route.called
    assert anthropic_route.called

