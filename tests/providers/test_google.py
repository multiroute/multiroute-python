import pytest
import respx
import httpx
import os
from multiroute.google import Client
from google.genai import types

@pytest.fixture
def client():
    # We use a dummy API key for the real client
    return Client(api_key="google-test-key")

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "multiroute-test-key")

@respx.mock
def test_generate_content_success(client):
    # Mock Multiroute (OpenAI format)
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "id": "chatcmpl-123",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello from Multiroute!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10
            }
        })
    )
    
    # Mock Google (should NOT be called)
    google_route = respx.post(url__regex=r"https://generativelanguage.googleapis.com/.*").mock(
        return_value=httpx.Response(200, json={})
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hi"
    )
    
    assert response.text == "Hello from Multiroute!"
    assert response.usage_metadata.prompt_token_count == 5
    assert multiroute_route.called
    assert not google_route.called
    
    # Check request headers
    request = multiroute_route.calls.last.request
    assert request.headers["Authorization"] == "Bearer multiroute-test-key"

@respx.mock
def test_generate_content_fallback(client):
    # Mock Multiroute failure
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "failed"})
    )
    
    # Mock Google success
    # Note: google-genai response format is complex, but we just need enough to satisfy the candidate extraction
    google_route = respx.post(url__regex=r"https://generativelanguage.googleapis.com/.*").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Google!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 3,
                "totalTokenCount": 6
            }
        })
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hi"
    )
    
    assert response.text == "Hello from Google!"
    assert multiroute_route.called
    assert google_route.called

@respx.mock
async def test_async_generate_content_success(client):
    # Async mock for Multiroute
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "Async Multiroute!"}}],
            "usage": {"total_tokens": 1}
        })
    )
    
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hi"
    )
    
    assert response.text == "Async Multiroute!"
    assert multiroute_route.called

@respx.mock
def test_no_multiroute_key(client, monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    
    multiroute_route = respx.post("https://api.multiroute.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    
    google_route = respx.post(url__regex=r"https://generativelanguage.googleapis.com/.*").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Direct Google!"}]}}]
        })
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hi"
    )
    
    assert response.text == "Direct Google!"
    assert not multiroute_route.called
    assert google_route.called
