from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, Union

import anthropic
import httpx
import openai
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

from multiroute.config import get_api_key, get_multiroute_base_url
from multiroute.providers import resolve_model

logger = logging.getLogger(__name__)


def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(e, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    if isinstance(
        e,
        (anthropic.InternalServerError, openai.InternalServerError),
    ):  # 5xx errors
        return True
    if isinstance(e, (anthropic.APITimeoutError, openai.APITimeoutError)):
        return True
    if isinstance(e, (anthropic.NotFoundError, openai.NotFoundError)):  # 404
        return True
    # Catch httpx generic errors that might happen when we bypass the client
    if isinstance(e, httpx.RequestError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code >= 500 or e.response.status_code == 404:
            return True
    return False


_shared_openai_client = None
_shared_async_openai_client = None


def _get_shared_openai_client() -> openai.OpenAI:
    global _shared_openai_client
    if _shared_openai_client is None:
        _shared_openai_client = openai.OpenAI(
            base_url=get_multiroute_base_url(),
            api_key=get_api_key(),
            max_retries=0,
        )
    return _shared_openai_client


def _get_shared_async_openai_client() -> openai.AsyncOpenAI:
    global _shared_async_openai_client
    if _shared_async_openai_client is None:
        _shared_async_openai_client = openai.AsyncOpenAI(
            base_url=get_multiroute_base_url(),
            api_key=get_api_key(),
            max_retries=0,
        )
    return _shared_async_openai_client


def _anthropic_to_openai_request(
    kwargs: dict[str, Any],
    base_url: str | None = None,
) -> dict[str, Any]:
    """Convert Anthropic messages request parameters to OpenAI chat/completions format."""
    openai_req = {}
    model = kwargs["model"]

    # Model
    if "model" in kwargs:
        openai_req["model"] = resolve_model(model, base_url)

    # System prompt -> role: system
    messages = []
    if kwargs.get("system"):
        # system could be string or list of text blocks
        sys_content = kwargs["system"]
        if isinstance(sys_content, list):
            # simplify list of text blocks to a single string for openai
            sys_text = "".join(b["text"] for b in sys_content if b["type"] == "text")
            messages.append({"role": "system", "content": sys_text})
        else:
            messages.append({"role": "system", "content": sys_content})

    # Messages
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            # Basic mapping, complex content blocks might need deeper extraction
            # Anthropic allows content to be string or list of blocks
            content = msg["content"]
            if isinstance(content, list):
                # Convert anthropic text blocks to openai text content
                openai_content = []
                tool_calls = []
                for block in content:
                    # Handle both dictionaries and Anthropic Pydantic models
                    b_type = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )

                    if b_type == "text":
                        b_text = (
                            block.get("text")
                            if isinstance(block, dict)
                            else getattr(block, "text", "")
                        )
                        openai_content.append({"type": "text", "text": b_text})
                    elif b_type == "image":
                        # Convert image block
                        b_source = (
                            block.get("source")
                            if isinstance(block, dict)
                            else getattr(block, "source", {})
                        )
                        b_mime = (
                            b_source.get("media_type")
                            if isinstance(b_source, dict)
                            else getattr(b_source, "media_type", "")
                        )
                        b_data = (
                            b_source.get("data")
                            if isinstance(b_source, dict)
                            else getattr(b_source, "data", "")
                        )
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{b_mime};base64,{b_data}"},
                            },
                        )
                    elif b_type == "tool_use":
                        # Assistant turn with tool call
                        b_id = (
                            block.get("id")
                            if isinstance(block, dict)
                            else getattr(block, "id", "")
                        )
                        b_name = (
                            block.get("name")
                            if isinstance(block, dict)
                            else getattr(block, "name", "")
                        )
                        b_input = (
                            block.get("input")
                            if isinstance(block, dict)
                            else getattr(block, "input", {})
                        )
                        tool_calls.append(
                            {
                                "id": b_id,
                                "type": "function",
                                "function": {
                                    "name": b_name,
                                    "arguments": json.dumps(b_input),
                                },
                            },
                        )
                    elif b_type == "tool_result":
                        # Tool result back to model
                        b_id = (
                            block.get("tool_use_id")
                            if isinstance(block, dict)
                            else getattr(block, "tool_use_id", "")
                        )
                        b_content = (
                            block.get("content", "")
                            if isinstance(block, dict)
                            else getattr(block, "content", "")
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": b_id,
                                "content": str(b_content),
                            },
                        )
                        continue  # Skip appending to current role's content

                if tool_calls:
                    # OpenAI assistant message with tool_calls
                    msg_obj = {"role": "assistant", "tool_calls": tool_calls}
                    # If there's text content, OpenAI allows it alongside tool_calls
                    # but it must be a string or a list of content parts.
                    if openai_content:
                        # Anthropic text blocks were already converted to OpenAI parts format
                        msg_obj["content"] = (
                            openai_content[0]["text"]
                            if len(openai_content) == 1
                            else openai_content
                        )
                    else:
                        # OpenAI requires 'content' to be present (can be null) even if tool_calls are present
                        msg_obj["content"] = None
                    messages.append(msg_obj)
                elif openai_content:
                    # If this message ONLY contained tool_result blocks (which were already handled above)
                    # and no text/image blocks, we don't want to append an empty user message.
                    # But if there are content parts, append them.
                    messages.append({"role": msg["role"], "content": openai_content})
                elif (
                    not tool_calls
                    and not openai_content
                    and any(m["role"] == "tool" for m in messages)
                ):
                    # Special case: If this turn ONLY contained tool_result blocks,
                    # those have already been appended to 'messages' as role: 'tool'.
                    # We do NOT append anything else for this turn.
                    pass
                elif not tool_calls and not openai_content:
                    # No content blocks at all (shouldn't happen in valid Anthropic requests)
                    pass
            else:
                messages.append({"role": msg["role"], "content": content})

    openai_req["messages"] = messages

    if "tools" in kwargs:
        openai_tools = []
        for tool in kwargs["tools"]:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                },
            )
        openai_req["tools"] = openai_tools

    if "tool_choice" in kwargs:
        tc = kwargs["tool_choice"]
        if tc.get("type") == "auto":
            openai_req["tool_choice"] = "auto"
        elif tc.get("type") == "any":
            openai_req["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            openai_req["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }

    if "max_tokens" in kwargs:
        openai_req["max_tokens"] = kwargs["max_tokens"]
    if "temperature" in kwargs:
        openai_req["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        openai_req["top_p"] = kwargs["top_p"]
    if "stop_sequences" in kwargs:
        openai_req["stop"] = kwargs["stop_sequences"]

    return openai_req


def _openai_to_anthropic_response(openai_resp: dict[str, Any]) -> Message:
    """Convert OpenAI chat/completions response to Anthropic Message type."""
    choice = openai_resp.get("choices", [{}])[0]
    message_data = choice.get("message", {})

    # Map finish reason
    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "stop":
        stop_reason = "end_turn"

    # Content
    content = message_data.get("content", "")
    content_blocks = [{"type": "text", "text": content}] if content else []

    # Tool calls
    tool_calls = message_data.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            # Safely parse arguments, they might be string or dict
            try:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}

            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc["function"]["name"],
                    "input": args,
                },
            )

    # Usage
    usage_data = openai_resp.get("usage", {})

    # Create the dictionary structure expected by Anthropic models
    msg_dict = {
        "id": openai_resp.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": openai_resp.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        },
    }

    return Message.construct(**msg_dict)


RawMessageStreamEvent = Union[
    RawMessageStartEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
]


def _openai_stream_to_anthropic_events(
    openai_stream: Any,
    model: str,
    message_id: str,
    input_tokens: int,
) -> Iterator[RawMessageStreamEvent]:
    """Translate an OpenAI Stream[ChatCompletionChunk] to Anthropic raw stream events."""
    # Emit message_start
    yield RawMessageStartEvent(
        type="message_start",
        message=Message.construct(
            id=message_id,
            type="message",
            role="assistant",
            content=[],
            model=model,
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        ),
    )
    # Emit content_block_start for text block 0
    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=TextBlock(type="text", text=""),
    )

    finish_reason = None
    output_tokens = 0

    for chunk in openai_stream:
        if not chunk.choices:
            # Usage-only chunk (some providers send this at the end)
            if hasattr(chunk, "usage") and chunk.usage:
                output_tokens = chunk.usage.completion_tokens or 0
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if delta and delta.content:
            yield RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text=delta.content),
            )

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        # Capture per-chunk usage if present
        if hasattr(chunk, "usage") and chunk.usage:
            output_tokens = chunk.usage.completion_tokens or output_tokens

    # Map finish_reason to Anthropic stop_reason
    stop_reason: str
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    yield RawContentBlockStopEvent(type="content_block_stop", index=0)
    yield RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason=stop_reason, stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=output_tokens),
    )
    yield RawMessageStopEvent(type="message_stop")


async def _openai_async_stream_to_anthropic_events(
    openai_stream: Any,
    model: str,
    message_id: str,
    input_tokens: int,
) -> AsyncIterator[RawMessageStreamEvent]:
    """Translate an OpenAI AsyncStream[ChatCompletionChunk] to Anthropic raw stream events."""
    yield RawMessageStartEvent(
        type="message_start",
        message=Message.construct(
            id=message_id,
            type="message",
            role="assistant",
            content=[],
            model=model,
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        ),
    )
    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=TextBlock(type="text", text=""),
    )

    finish_reason = None
    output_tokens = 0

    async for chunk in openai_stream:
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage:
                output_tokens = chunk.usage.completion_tokens or 0
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if delta and delta.content:
            yield RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text=delta.content),
            )

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if hasattr(chunk, "usage") and chunk.usage:
            output_tokens = chunk.usage.completion_tokens or output_tokens

    stop_reason: str
    if finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"

    yield RawContentBlockStopEvent(type="content_block_stop", index=0)
    yield RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason=stop_reason, stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=output_tokens),
    )
    yield RawMessageStopEvent(type="message_stop")


class _SyncAnthropicStream:
    """Wraps an OpenAI Stream[ChatCompletionChunk] and yields Anthropic RawMessageStreamEvents."""

    def __init__(
        self,
        openai_stream: Any,
        model: str,
        message_id: str = "multiroute-stream",
        input_tokens: int = 0,
    ) -> None:
        self._openai_stream = openai_stream
        self._iterator = _openai_stream_to_anthropic_events(
            openai_stream,
            model,
            message_id,
            input_tokens,
        )

    def __iter__(self) -> Iterator[RawMessageStreamEvent]:
        return self._iterator

    def __next__(self) -> RawMessageStreamEvent:
        return next(self._iterator)

    def __enter__(self) -> _SyncAnthropicStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self._openai_stream, "close"):
            self._openai_stream.close()


class _AsyncAnthropicStream:
    """Wraps an OpenAI AsyncStream[ChatCompletionChunk] and yields Anthropic events."""

    def __init__(
        self,
        openai_stream: Any,
        model: str,
        message_id: str = "multiroute-stream",
        input_tokens: int = 0,
    ) -> None:
        self._openai_stream = openai_stream
        self._iterator = _openai_async_stream_to_anthropic_events(
            openai_stream,
            model,
            message_id,
            input_tokens,
        )

    def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        return self._iterator

    async def __anext__(self) -> RawMessageStreamEvent:
        return await self._iterator.__anext__()

    async def __aenter__(self) -> _AsyncAnthropicStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if hasattr(self._openai_stream, "aclose"):
            await self._openai_stream.aclose()


class MultirouteMessages(Messages):
    def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_api_key()

        if not mr_api_key:
            return super().create(**kwargs)

        stream = kwargs.get("stream", False)

        try:
            openai_req = _anthropic_to_openai_request(
                kwargs,
                str(self._client.base_url),
            )

            client = _get_shared_openai_client().with_options(
                base_url=get_multiroute_base_url(),
                api_key=mr_api_key,
                timeout=self._client.timeout,
            )

            if stream:
                openai_req["stream"] = True
                openai_stream = client.chat.completions.create(**openai_req)
                return _SyncAnthropicStream(
                    openai_stream,
                    model=kwargs.get("model", ""),
                )
            openai_resp_obj = client.chat.completions.create(**openai_req)
            openai_resp = openai_resp_obj.model_dump()
            return _openai_to_anthropic_response(openai_resp)
        except Exception as e:
            if _is_multiroute_error(e):
                # Fallback to the real anthropic create using original parameters
                return super().create(**kwargs)
            raise


class AsyncMultirouteMessages(AsyncMessages):
    async def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_api_key()

        if not mr_api_key:
            return await super().create(**kwargs)

        stream = kwargs.get("stream", False)

        try:
            openai_req = _anthropic_to_openai_request(
                kwargs,
                str(self._client.base_url),
            )

            client = _get_shared_async_openai_client().with_options(
                base_url=get_multiroute_base_url(),
                api_key=mr_api_key,
                timeout=self._client.timeout,
            )

            if stream:
                openai_req["stream"] = True
                openai_stream = await client.chat.completions.create(**openai_req)
                return _AsyncAnthropicStream(
                    openai_stream,
                    model=kwargs.get("model", ""),
                )
            openai_resp_obj = await client.chat.completions.create(**openai_req)
            openai_resp = openai_resp_obj.model_dump()
            return _openai_to_anthropic_response(openai_resp)
        except Exception as e:
            if _is_multiroute_error(e):
                return await super().create(**kwargs)
            raise


class Anthropic(anthropic.Anthropic):
    def __init__(self, *args, **kwargs):
        self.multiroute_api_key = (
            kwargs.pop("multiroute_api_key", None) or get_api_key()
        )
        super().__init__(*args, **kwargs)
        if not self.multiroute_api_key:
            logger.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to Anthropic "
                "without Multiroute high-availability routing.",
            )
        self.messages = MultirouteMessages(self)


class AsyncAnthropic(anthropic.AsyncAnthropic):
    def __init__(self, *args, **kwargs):
        self.multiroute_api_key = (
            kwargs.pop("multiroute_api_key", None) or get_api_key()
        )
        super().__init__(*args, **kwargs)
        if not self.multiroute_api_key:
            logger.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to Anthropic "
                "without Multiroute high-availability routing.",
            )
        self.messages = AsyncMultirouteMessages(self)
