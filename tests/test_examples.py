"""
Verification tests for all example files.

These tests mock the HTTP layer with respx to confirm each example's code paths
work correctly — streaming, tool calls, sync and async — without live API calls.
"""

import json
import sys
import os
import pytest
import respx
import httpx
from httpx._content import AsyncIteratorByteStream, IteratorByteStream
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Shared mock bodies
# ---------------------------------------------------------------------------

_OPENAI_BASIC_JSON = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_OPENAI_TOOL_CALL_JSON = {
    "id": "chatcmpl-t1",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
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
                            "arguments": '{"location": "Boston, MA"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

_OPENAI_FINAL_JSON = {
    "id": "chatcmpl-t2",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "It's sunny and 22°C in Boston.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
}

# OpenAI SSE stream (two text chunks + done)
_OPENAI_SSE = (
    b'data: {"id":"s1","object":"chat.completion.chunk","model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
    b'data: {"id":"s1","object":"chat.completion.chunk","model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n'
    b'data: {"id":"s1","object":"chat.completion.chunk","model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)

# Anthropic tool-call response (via OpenAI proxy translation)
_ANTHROPIC_TOOL_CALL_JSON = {
    "id": "chatcmpl-a1",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Boston, MA"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

# Google SSE stream
_GOOGLE_SSE = (
    b'data: {"id":"g1","object":"chat.completion.chunk","model":"gemini-2.0-flash",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
    b'data: {"id":"g1","object":"chat.completion.chunk","model":"gemini-2.0-flash",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)

# Native Anthropic SSE
_ANTHROPIC_NATIVE_SSE = (
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

MULTIROUTE_URL = "https://api.multiroute.ai/openai/v1/chat/completions"
LITELLM_MULTIROUTE_URL = "https://api.multiroute.ai/v1/chat/completions"


async def _aiter(chunks):
    for c in chunks:
        yield c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_multiroute_key(monkeypatch):
    monkeypatch.setenv("MULTIROUTE_API_KEY", "test-multiroute-key")


# ---------------------------------------------------------------------------
# OpenAI examples
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_usage_sync_basic():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_OPENAI_BASIC_JSON)
    )

    from multiroute.openai import OpenAI

    client = OpenAI(api_key="test")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    assert response.choices[0].message.content == "Hello there!"


@respx.mock
def test_openai_usage_sync_streaming():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_OPENAI_SSE]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.openai import OpenAI

    client = OpenAI(api_key="test")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
        stream=True,
    )
    text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text == "Hello world"


@respx.mock
async def test_openai_usage_async_basic():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_OPENAI_BASIC_JSON)
    )

    from multiroute.openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    assert response.choices[0].message.content == "Hello there!"


@respx.mock
async def test_openai_usage_async_streaming():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=AsyncIteratorByteStream(_aiter([_OPENAI_SSE])),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test")
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
        stream=True,
    )
    text = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text == "Hello world"


# ---------------------------------------------------------------------------
# OpenAI tools examples
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_tools_sync_two_turn():
    route = respx.post(MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_OPENAI_TOOL_CALL_JSON),
        httpx.Response(200, json=_OPENAI_FINAL_JSON),
    ]

    from multiroute.openai import OpenAI

    client = OpenAI(api_key="test")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "Weather in Boston?"}]
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    message = response.choices[0].message
    assert message.tool_calls is not None
    assert message.tool_calls[0].function.name == "get_weather"

    messages.append(message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": message.tool_calls[0].id,
            "name": "get_weather",
            "content": "Sunny, 22°C",
        }
    )

    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assert (
        "Boston" in final.choices[0].message.content
        or "sunny" in final.choices[0].message.content.lower()
    )


@respx.mock
async def test_openai_tools_async_two_turn():
    route = respx.post(MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_OPENAI_TOOL_CALL_JSON),
        httpx.Response(200, json=_OPENAI_FINAL_JSON),
    ]

    from multiroute.openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="test")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "Weather in London?"}]
    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    message = response.choices[0].message
    assert message.tool_calls is not None

    messages.append(message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": message.tool_calls[0].id,
            "name": "get_weather",
            "content": "Cloudy, 14°C",
        }
    )

    final = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    assert final.choices[0].message.content is not None


# ---------------------------------------------------------------------------
# Anthropic examples
# ---------------------------------------------------------------------------


@respx.mock
def test_anthropic_usage_sync_basic():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_OPENAI_BASIC_JSON)
    )

    from multiroute.anthropic import Anthropic

    client = Anthropic(api_key="test", max_retries=0)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    assert response.content[0].text == "Hello there!"


@respx.mock
def test_anthropic_usage_sync_streaming():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_OPENAI_SSE]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.anthropic import Anthropic

    client = Anthropic(api_key="test", max_retries=0)
    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello!"}],
        stream=True,
    )
    events = list(stream)
    deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert "Hello" in "".join(deltas)


@respx.mock
async def test_anthropic_usage_async_streaming():
    respx.post(MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=AsyncIteratorByteStream(_aiter([_OPENAI_SSE])),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key="test", max_retries=0)
    stream = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello!"}],
        stream=True,
    )
    events = []
    async for event in stream:
        events.append(event)
    deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert "Hello" in "".join(deltas)


# ---------------------------------------------------------------------------
# Anthropic tools examples
# ---------------------------------------------------------------------------


@respx.mock
def test_anthropic_tools_sync_two_turn():
    route = respx.post(MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_ANTHROPIC_TOOL_CALL_JSON),
        httpx.Response(200, json=_OPENAI_BASIC_JSON),
    ]

    from multiroute.anthropic import Anthropic

    client = Anthropic(api_key="test", max_retries=0)
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "Weather in Boston?"}]
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            assert block.name == "get_weather"
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Sunny, 22°C",
                }
            )

    assert len(tool_results) == 1

    messages.append({"role": "user", "content": tool_results})
    final = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
    )
    assert final.content[0].text == "Hello there!"


@respx.mock
async def test_anthropic_tools_async_two_turn():
    route = respx.post(MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_ANTHROPIC_TOOL_CALL_JSON),
        httpx.Response(200, json=_OPENAI_BASIC_JSON),
    ]

    from multiroute.anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key="test", max_retries=0)
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    messages = [{"role": "user", "content": "Weather in London?"}]
    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Cloudy, 14°C",
                }
            )

    assert len(tool_results) == 1
    messages.append({"role": "user", "content": tool_results})

    final = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
    )
    assert final.content[0].text == "Hello there!"


# ---------------------------------------------------------------------------
# Google examples
# ---------------------------------------------------------------------------

_GOOGLE_BASIC_JSON = {
    "id": "chatcmpl-g1",
    "object": "chat.completion",
    "model": "gemini-2.0-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Bonjour!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

_GOOGLE_TOOL_CALL_JSON = {
    "id": "chatcmpl-g2",
    "object": "chat.completion",
    "model": "gemini-2.0-flash",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "get_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Boston"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

GOOGLE_MULTIROUTE_URL = "https://api.multiroute.ai/openai/v1/chat/completions"


@respx.mock
def test_google_usage_sync_basic():
    respx.post(GOOGLE_MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_GOOGLE_BASIC_JSON)
    )

    from multiroute.google import Client

    client = Client(api_key="test-google")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Say hello!"
    )
    assert response.text == "Bonjour!"


@respx.mock
def test_google_usage_sync_streaming():
    respx.post(GOOGLE_MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_GOOGLE_SSE]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.google import Client

    client = Client(api_key="test-google")
    chunks = list(
        client.models.generate_content_stream(model="gemini-2.0-flash", contents="Hi!")
    )
    text = "".join(
        c.text
        for c in chunks
        if c.candidates and c.candidates[0].content.parts and c.text
    )
    assert "Hello" in text


@respx.mock
async def test_google_usage_async_basic():
    respx.post(GOOGLE_MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_GOOGLE_BASIC_JSON)
    )

    from multiroute.google import Client

    client = Client(api_key="test-google")
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash", contents="Say hello!"
    )
    assert response.text == "Bonjour!"


@respx.mock
async def test_google_usage_async_streaming():
    respx.post(GOOGLE_MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=AsyncIteratorByteStream(_aiter([_GOOGLE_SSE])),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.google import Client

    client = Client(api_key="test-google")
    stream = await client.aio.models.generate_content_stream(
        model="gemini-2.0-flash", contents="Hi!"
    )
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    text = "".join(c.text for c in chunks if c.text)
    assert "Hello" in text


# ---------------------------------------------------------------------------
# Google tools examples
# ---------------------------------------------------------------------------


@respx.mock
def test_google_tools_sync_two_turn():
    route = respx.post(GOOGLE_MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_GOOGLE_TOOL_CALL_JSON),
        httpx.Response(200, json=_GOOGLE_BASIC_JSON),
    ]

    from multiroute.google import Client
    from google.genai import types

    def get_weather(location: str) -> str:
        """Get the current weather."""
        return f"Sunny in {location}"

    client = Client(api_key="test-google")
    contents = ["What's the weather in Boston?"]

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    contents.append(response.candidates[0].content)

    assert response.function_calls is not None
    assert len(response.function_calls) == 1
    fc = response.function_calls[0]
    assert fc.name == "get_weather"

    tool_responses = [
        types.Part(
            function_response=types.FunctionResponse(
                name=fc.name, response={"result": "Sunny, 22°C"}
            )
        )
    ]
    contents.append(types.Content(role="user", parts=tool_responses))

    final = client.models.generate_content(model="gemini-2.0-flash", contents=contents)
    assert final.text == "Bonjour!"


@respx.mock
async def test_google_tools_async_two_turn():
    route = respx.post(GOOGLE_MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_GOOGLE_TOOL_CALL_JSON),
        httpx.Response(200, json=_GOOGLE_BASIC_JSON),
    ]

    from multiroute.google import Client
    from google.genai import types

    def get_weather(location: str) -> str:
        """Get the current weather."""
        return f"Sunny in {location}"

    client = Client(api_key="test-google")
    contents = ["What's the weather in London?"]

    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    contents.append(response.candidates[0].content)

    assert response.function_calls is not None
    fc = response.function_calls[0]

    tool_responses = [
        types.Part(
            function_response=types.FunctionResponse(
                name=fc.name, response={"result": "Cloudy, 14°C"}
            )
        )
    ]
    contents.append(types.Content(role="user", parts=tool_responses))

    final = await client.aio.models.generate_content(
        model="gemini-2.0-flash", contents=contents
    )
    assert final.text == "Bonjour!"


# ---------------------------------------------------------------------------
# LiteLLM examples
# ---------------------------------------------------------------------------

_LITELLM_BASIC_JSON = {
    "id": "chatcmpl-l1",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

_LITELLM_TOOL_CALL_JSON = {
    "id": "chatcmpl-l2",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_lit1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Boston, MA"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


@respx.mock
def test_litellm_sync_basic():
    respx.post(LITELLM_MULTIROUTE_URL).mock(
        return_value=httpx.Response(200, json=_LITELLM_BASIC_JSON)
    )

    from multiroute.litellm import completion

    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    assert response.choices[0].message.content == "Hello!"


@respx.mock
def test_litellm_sync_streaming():
    respx.post(LITELLM_MULTIROUTE_URL).mock(
        return_value=httpx.Response(
            200,
            stream=IteratorByteStream([_OPENAI_SSE]),
            headers={"Content-Type": "text/event-stream"},
        )
    )

    from multiroute.litellm import completion

    stream = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count!"}],
        stream=True,
    )
    text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            text += delta.content
    assert "Hello" in text


@respx.mock
def test_litellm_sync_tools_two_turn():
    route = respx.post(LITELLM_MULTIROUTE_URL)
    route.side_effect = [
        httpx.Response(200, json=_LITELLM_TOOL_CALL_JSON),
        httpx.Response(200, json=_LITELLM_BASIC_JSON),
    ]

    from multiroute.litellm import completion

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "Weather in Boston?"}]
    response = completion(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    assert response_message.tool_calls is not None
    tc = response_message.tool_calls[0]
    assert tc.function.name == "get_current_weather"

    messages.append(
        {
            "tool_call_id": tc.id,
            "role": "tool",
            "name": tc.function.name,
            "content": json.dumps({"temperature": "72", "unit": "fahrenheit"}),
        }
    )

    second = completion(model="gpt-4o-mini", messages=messages)
    assert second.choices[0].message.content == "Hello!"
