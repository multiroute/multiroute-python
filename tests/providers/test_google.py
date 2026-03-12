import json
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
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Multiroute!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
            },
        )
    )

    # Mock Google (should NOT be called)
    google_route = respx.post(
        url__regex=r"https://generativelanguage.googleapis.com/.*"
    ).mock(return_value=httpx.Response(200, json={}))

    response = client.models.generate_content(model="gemini-2.0-flash", contents="Hi")

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
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "failed"})
    )

    # Mock Google success
    # Note: google-genai response format is complex, but we just need enough to satisfy the candidate extraction
    google_route = respx.post(
        url__regex=r"https://generativelanguage.googleapis.com/.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello from Google!"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 6,
                },
            },
        )
    )

    response = client.models.generate_content(model="gemini-2.0-flash", contents="Hi")

    assert response.text == "Hello from Google!"
    assert multiroute_route.called
    assert google_route.called


@respx.mock
async def test_async_generate_content_success(client):
    # Async mock for Multiroute
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Async Multiroute!"}}],
                "usage": {"total_tokens": 1},
            },
        )
    )

    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash", contents="Hi"
    )

    assert response.text == "Async Multiroute!"
    assert multiroute_route.called


@respx.mock
def test_no_multiroute_key(client, monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )

    google_route = respx.post(
        url__regex=r"https://generativelanguage.googleapis.com/.*"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Direct Google!"}]}}]},
        )
    )

    response = client.models.generate_content(model="gemini-2.0-flash", contents="Hi")

    assert response.text == "Direct Google!"
    assert not multiroute_route.called
    assert google_route.called


@respx.mock
def test_tools_request_translation(client):
    """Tools in GenerateContentConfig should be translated into OpenAI tools format."""
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-gt-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "total_tokens": 23,
                },
            },
        )
    )

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city to get weather for.
        """
        return f"Sunny in {location}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What is the weather in Paris?",
        config=types.GenerateContentConfig(tools=[get_weather]),
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)

    # Tools should be present and in OpenAI format
    assert "tools" in req_json
    assert len(req_json["tools"]) >= 1
    tool = req_json["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    # Parameters should have lowercase types
    params = tool["function"]["parameters"]
    assert "properties" in params
    # No UPPERCASE type values (all lowercased)
    params_str = json.dumps(params)
    assert "OBJECT" not in params_str
    assert "STRING" not in params_str

    # Response should contain a function_call part
    assert response.candidates is not None
    assert len(response.candidates) > 0
    parts = response.candidates[0].content.parts
    fc_parts = [p for p in parts if p.function_call is not None]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call.name == "get_weather"
    assert fc_parts[0].function_call.args == {"location": "Paris"}


@respx.mock
def test_function_response_contents_translation(client):
    """Contents with function_response parts should become OpenAI tool-role messages."""
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-fr-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The weather in Paris is sunny.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                },
            },
        )
    )

    # Multi-turn: user asks, model calls tool, user provides tool result
    contents = [
        types.Content(
            role="user", parts=[types.Part(text="What is the weather in Paris?")]
        ),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather", args={"location": "Paris"}
                    )
                )
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_weather", response={"result": "Sunny, 22°C"}
                    )
                )
            ],
        ),
    ]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    messages = req_json["messages"]

    # There should be a tool-role message for the function response
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "get_weather"
    assert json.loads(tool_msgs[0]["content"]) == {"result": "Sunny, 22°C"}

    # There should be an assistant message with tool_calls for the function call
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    assert "tool_calls" in assistant_msgs[0]
    assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    assert response.text == "The weather in Paris is sunny."


@respx.mock
def test_mixed_text_and_function_call_translation(client):
    """Contents with both text and function_call should preserve both when translating to OpenAI."""
    multiroute_route = respx.post("https://api.multiroute.ai/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-mixed-g1",
                "choices": [
                    {"message": {"content": "Confirmed."}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 10},
            },
        )
    )

    contents = [
        types.Content(
            role="model",
            parts=[
                types.Part(text="Let me check that for you."),
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather", args={"location": "London"}
                    )
                ),
            ],
        )
    ]

    client.models.generate_content(model="gemini-2.0-flash", contents=contents)

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    assistant_msg = req_json["messages"][0]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Let me check that for you."
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
