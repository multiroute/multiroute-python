import json
import os
from typing import Any, Dict

import anthropic
import httpx
import openai
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import Message

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/v1"


def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(e, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    if isinstance(
        e, (anthropic.InternalServerError, openai.InternalServerError)
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
            base_url=MULTIROUTE_BASE_URL,
            api_key=os.environ.get("MULTIROUTE_API_KEY") or "dummy",
            max_retries=0,
        )
    return _shared_openai_client


def _get_shared_async_openai_client() -> openai.AsyncOpenAI:
    global _shared_async_openai_client
    if _shared_async_openai_client is None:
        _shared_async_openai_client = openai.AsyncOpenAI(
            base_url=MULTIROUTE_BASE_URL,
            api_key=os.environ.get("MULTIROUTE_API_KEY") or "dummy",
            max_retries=0,
        )
    return _shared_async_openai_client


def _anthropic_to_openai_request(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic messages request parameters to OpenAI chat/completions format."""
    openai_req = {}
    model = kwargs["model"]

    # Model
    if "model" in kwargs:
        openai_req["model"] = model

    # System prompt -> role: system
    messages = []
    if "system" in kwargs and kwargs["system"]:
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
                            }
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
                            }
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
                            }
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
                }
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


def _openai_to_anthropic_response(openai_resp: Dict[str, Any]) -> Message:
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
                }
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


class MultirouteMessages(Messages):
    def create(self, **kwargs) -> Message:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return super().create(**kwargs)

        # Save original API URL and client behavior
        original_base_url = self._client.base_url

        try:
            openai_req = _anthropic_to_openai_request(kwargs)

            client = _get_shared_openai_client().with_options(
                api_key=os.environ.get("MULTIROUTE_API_KEY"),
                timeout=self._client.timeout,
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
    async def create(self, **kwargs) -> Message:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await super().create(**kwargs)

        original_base_url = self._client.base_url

        try:
            openai_req = _anthropic_to_openai_request(kwargs)

            client = _get_shared_async_openai_client().with_options(
                api_key=os.environ.get("MULTIROUTE_API_KEY"),
                timeout=self._client.timeout,
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
        super().__init__(*args, **kwargs)
        self.messages = MultirouteMessages(self)


class AsyncAnthropic(anthropic.AsyncAnthropic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = AsyncMultirouteMessages(self)
