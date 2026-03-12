import pytest
import respx
import httpx
from httpx._content import AsyncIteratorByteStream, IteratorByteStream
import json
import openai
import anthropic
from multiroute.anthropic import Anthropic, AsyncAnthropic
from multiroute.anthropic.client import MULTIROUTE_BASE_URL


async def aiter_bytes(chunks: list):
    """Async generator that yields byte chunks — for use with AsyncIteratorByteStream."""
    for chunk in chunks:
        yield chunk


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
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4o",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello there!"},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={})
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
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
    assert req_json["model"] == "anthropic/claude-3-5-sonnet-20241022"
    assert req_json["messages"] == [{"role": "user", "content": "Hello!"}]
    assert req_json["max_tokens"] == 100


def import_json(content):
    import json

    return json.loads(content)


@respx.mock
def test_messages_fallback_500(client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Fallback response!"}],
                "usage": {"input_tokens": 9, "output_tokens": 12},
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
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
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [
                    {
                        "type": "text",
                        "text": "Fallback response after connection error!",
                    }
                ],
                "usage": {"input_tokens": 9, "output_tokens": 12},
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
    )

    assert response.content[0].text == "Fallback response after connection error!"
    assert multiroute_route.called
    assert anthropic_route.called


@respx.mock
async def test_async_messages_fallback_500(async_client):
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Async Fallback response!"}],
                "usage": {"input_tokens": 9, "output_tokens": 12},
            },
        )
    )

    response = await async_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
    )

    assert response.content[0].text == "Async Fallback response!"
    assert multiroute_route.called
    assert anthropic_route.called


@respx.mock
def test_tools_request_translation(client):
    """Tools in Anthropic format should be translated to OpenAI format in the proxy request."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-1",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "London"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "What's the weather?"}],
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)

    # Verify tools were translated to OpenAI format
    assert "tools" in req_json
    assert len(req_json["tools"]) == 1
    tool = req_json["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["description"] == "Get the current weather"
    assert tool["function"]["parameters"] == {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    }

    # Verify response was translated back to Anthropic format
    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    tool_block = response.content[0]
    assert tool_block.type == "tool_use"
    assert tool_block.id == "call_abc"
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"location": "London"}


@respx.mock
def test_tool_choice_translation(client):
    """tool_choice variants should be translated correctly to OpenAI format."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-tc-1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            },
        )
    )

    # Test tool_choice "any" -> "required"
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
        tools=[
            {
                "name": "say_hi",
                "description": "Say hi",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_choice={"type": "any"},
    )

    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["tool_choice"] == "required"

    # Test tool_choice "auto" -> "auto"
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
        tools=[
            {
                "name": "say_hi",
                "description": "Say hi",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_choice={"type": "auto"},
    )

    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["tool_choice"] == "auto"

    # Test tool_choice "tool" -> specific function
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
        tools=[
            {
                "name": "say_hi",
                "description": "Say hi",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        tool_choice={"type": "tool", "name": "say_hi"},
    )

    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["tool_choice"] == {
        "type": "function",
        "function": {"name": "say_hi"},
    }


@respx.mock
def test_tool_result_message_translation(client):
    """tool_result content blocks in user messages should become OpenAI tool-role messages."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-tr-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The weather is sunny.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 8,
                    "total_tokens": 38,
                },
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_abc",
                        "name": "get_weather",
                        "input": {"location": "London"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": "Sunny, 22°C",
                    }
                ],
            },
        ],
        max_tokens=100,
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    messages = req_json["messages"]

    # The tool_result should produce a tool-role message
    tool_msg = next((m for m in messages if m.get("role") == "tool"), None)
    assert tool_msg is not None
    assert tool_msg["tool_call_id"] == "call_abc"
    assert tool_msg["content"] == "Sunny, 22°C"

    # No bare user message emitted for that turn (only the tool message)
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "What's the weather?"

    assert response.content[0].text == "The weather is sunny."


@respx.mock
def test_messages_no_multiroute_key(client, monkeypatch):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Direct Anthropic!"}],
                "usage": {"input_tokens": 9, "output_tokens": 12},
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
    )

    assert response.content[0].text == "Direct Anthropic!"
    assert not multiroute_route.called
    assert anthropic_route.called


@respx.mock
def test_mixed_text_and_tool_use_translation(client):
    """A message with both text and tool_use should preserve both when translating to OpenAI."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-mixed-1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Acknowledged."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )
    )

    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will check the weather now."},
                    {
                        "type": "tool_use",
                        "id": "call_mixed",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    },
                ],
            }
        ],
        max_tokens=100,
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    assistant_msg = req_json["messages"][0]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "I will check the weather now."
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Async no-key path
# ---------------------------------------------------------------------------

_ANTHROPIC_SUCCESS_JSON = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "content": [{"type": "text", "text": "Direct Anthropic async!"}],
    "usage": {"input_tokens": 9, "output_tokens": 12},
}


@respx.mock
async def test_async_messages_no_multiroute_key(async_client, monkeypatch):
    """AsyncAnthropic calls native Anthropic directly when no key is set."""
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)

    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_SUCCESS_JSON)
    )

    response = await async_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
    )

    assert response.content[0].text == "Direct Anthropic async!"
    assert not multiroute_route.called
    assert anthropic_route.called


# ---------------------------------------------------------------------------
# Async connection-error fallback
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_messages_fallback_connection_error(async_client):
    """AsyncAnthropic falls back to native when proxy raises a connection error."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_conn_async",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [
                    {"type": "text", "text": "Async fallback after connect error!"}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 8},
            },
        )
    )

    response = await async_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
    )

    assert response.content[0].text == "Async fallback after connect error!"
    assert multiroute_route.called
    assert anthropic_route.called


# ---------------------------------------------------------------------------
# System prompt translation
# ---------------------------------------------------------------------------


@respx.mock
def test_system_prompt_translation(client):
    """A 'system' kwarg should appear as a system-role message in the proxy request."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-sys",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Got it."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
        )
    )

    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50,
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    messages = req_json["messages"]

    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are a helpful assistant."


@respx.mock
def test_list_system_prompt_translation(client):
    """A list-style 'system' kwarg (text blocks) should be joined into a single string."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-sys-list",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Ok."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                },
            },
        )
    )

    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=[
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": " Be concise."},
        ],
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=50,
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    messages = req_json["messages"]

    system_msgs = [m for m in messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "You are helpful. Be concise."


# ---------------------------------------------------------------------------
# temperature / top_p / stop_sequences translation
# ---------------------------------------------------------------------------


@respx.mock
def test_sampling_params_translation(client):
    """temperature, top_p, and stop_sequences are forwarded to the proxy request."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-sampling",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )
    )

    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=50,
        temperature=0.3,
        top_p=0.9,
        stop_sequences=["STOP", "END"],
    )

    assert multiroute_route.called
    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["temperature"] == 0.3
    assert req_json["top_p"] == 0.9
    assert req_json["stop"] == ["STOP", "END"]


# ---------------------------------------------------------------------------
# finish_reason=length -> stop_reason=max_tokens
# ---------------------------------------------------------------------------


@respx.mock
def test_finish_reason_length_maps_to_max_tokens(client):
    """finish_reason='length' from OpenAI should map to stop_reason='max_tokens' in Anthropic."""
    respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-len",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "truncated..."},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60,
                },
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Tell me a long story"}],
        max_tokens=50,
    )

    assert response.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# Non-multiroute errors are re-raised
# ---------------------------------------------------------------------------


@respx.mock
def test_messages_non_multiroute_error_reraised(client):
    """A 401 authentication error from the proxy should be re-raised, not swallowed."""
    respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {"message": "Invalid API key", "type": "authentication_error"}
            },
        )
    )

    with pytest.raises(openai.AuthenticationError):
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=50,
        )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

# SSE data for a simple two-chunk stream followed by [DONE]
_SSE_CHUNK_1 = (
    b'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},'
    b'"finish_reason":null}]}\n\n'
)
_SSE_CHUNK_2 = (
    b'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
)
_SSE_CHUNK_FINAL = (
    b'data: {"id":"chatcmpl-s1","object":"chat.completion.chunk","model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
)
_SSE_DONE = b"data: [DONE]\n\n"

_ANTHROPIC_SSE_BODY = _SSE_CHUNK_1 + _SSE_CHUNK_2 + _SSE_CHUNK_FINAL + _SSE_DONE

# Native Anthropic SSE format for fallback tests (when proxy fails and native is called)
_NATIVE_ANTHROPIC_SSE_BODY = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant",'
    b'"content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,'
    b'"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
    b"event: content_block_start\n"
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    b"event: content_block_delta\n"
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
    b"event: content_block_stop\n"
    b'data: {"type":"content_block_stop","index":0}\n\n'
    b"event: message_delta\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    b'"usage":{"output_tokens":1}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


@respx.mock
def test_messages_stream_success(client):
    """stream=True routes through the proxy and translates OpenAI SSE to Anthropic events."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_ANTHROPIC_SSE_BODY]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={})
    )

    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        stream=True,
    )

    # Verify proxy was called and native was NOT called
    assert multiroute_route.called
    assert not anthropic_route.called

    # Verify stream=true was sent to the proxy
    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["stream"] is True

    # Collect events
    events = list(stream)
    event_types = [e.type for e in events]

    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types

    # Check text deltas accumulate to "Hello world"
    text_deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert "".join(text_deltas) == "Hello world"

    # Check stop reason
    msg_delta = next(e for e in events if e.type == "message_delta")
    assert msg_delta.delta.stop_reason == "end_turn"


@respx.mock
def test_messages_stream_fallback_500(client):
    """stream=True falls back to native Anthropic when the proxy returns 500."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_NATIVE_ANTHROPIC_SSE_BODY]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        stream=True,
    )

    assert multiroute_route.called
    assert anthropic_route.called

    # The native Anthropic stream yields RawMessageStreamEvent objects
    events = list(stream)
    assert len(events) > 0


@respx.mock
def test_messages_stream_fallback_connection_error(client):
    """stream=True falls back to native Anthropic when the proxy raises a connection error."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_NATIVE_ANTHROPIC_SSE_BODY]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        stream=True,
    )

    assert multiroute_route.called
    assert anthropic_route.called
    events = list(stream)
    assert len(events) > 0


@respx.mock
async def test_async_messages_stream_success(async_client):
    """Async stream=True routes through the proxy and translates OpenAI SSE to Anthropic events."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            stream=AsyncIteratorByteStream(aiter_bytes([_ANTHROPIC_SSE_BODY])),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={})
    )

    stream = await async_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        stream=True,
    )

    assert multiroute_route.called
    assert not anthropic_route.called

    req_json = json.loads(multiroute_route.calls.last.request.content)
    assert req_json["stream"] is True

    events = []
    async for event in stream:
        events.append(event)

    event_types = [e.type for e in events]
    assert "message_start" in event_types
    assert "content_block_delta" in event_types
    assert "message_stop" in event_types

    text_deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert "".join(text_deltas) == "Hello world"


@respx.mock
async def test_async_messages_stream_fallback_500(async_client):
    """Async stream=True falls back to native Anthropic when proxy returns 500."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            500, json={"error": {"message": "Internal Server Error"}}
        )
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            stream=AsyncIteratorByteStream(aiter_bytes([_NATIVE_ANTHROPIC_SSE_BODY])),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    stream = await async_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=100,
        stream=True,
    )

    assert multiroute_route.called
    assert anthropic_route.called

    events = []
    async for event in stream:
        events.append(event)
    assert len(events) > 0


def test_no_multiroute_key_warns(monkeypatch, caplog):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    import logging

    with caplog.at_level(logging.ERROR):
        Anthropic(api_key="test-key")
    assert "MULTIROUTE_API_KEY is not set" in caplog.text


def test_async_no_multiroute_key_warns(monkeypatch, caplog):
    monkeypatch.delenv("MULTIROUTE_API_KEY", raising=False)
    import logging

    with caplog.at_level(logging.ERROR):
        AsyncAnthropic(api_key="test-key")
    assert "MULTIROUTE_API_KEY is not set" in caplog.text


# ---------------------------------------------------------------------------
# 404 fallback
# ---------------------------------------------------------------------------


@respx.mock
def test_messages_fallback_404(client):
    """A 404 from the proxy should trigger fallback to native Anthropic."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, json={"detail": "Not Found"})
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_404fb",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "404 Fallback!"}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50,
    )

    assert response.content[0].text == "404 Fallback!"
    assert multiroute_route.called
    assert anthropic_route.called


# ---------------------------------------------------------------------------
# Timeout fallback
# ---------------------------------------------------------------------------


@respx.mock
def test_messages_fallback_timeout(client):
    """An httpx.TimeoutException from the proxy should trigger fallback to native Anthropic."""
    multiroute_route = respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_timeout",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "Timeout Fallback!"}],
                "usage": {"input_tokens": 5, "output_tokens": 5},
            },
        )
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50,
    )

    assert response.content[0].text == "Timeout Fallback!"
    assert multiroute_route.called
    assert anthropic_route.called


# ---------------------------------------------------------------------------
# Async non-multiroute errors are re-raised
# ---------------------------------------------------------------------------


@respx.mock
async def test_async_messages_non_multiroute_error_reraised(async_client):
    """A 401 from the proxy in async mode should be re-raised, not swallowed."""
    respx.post(f"{MULTIROUTE_BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )
    )

    with pytest.raises(openai.AuthenticationError):
        await async_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=50,
        )
