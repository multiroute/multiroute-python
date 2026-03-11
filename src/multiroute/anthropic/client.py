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
                for block in content:
                    if block["type"] == "text":
                        openai_content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        # Convert image block
                        # Anthropic uses source: {type: "base64", media_type: "...", data: "..."}
                        # OpenAI uses image_url: {"url": "data:image/...;base64,..."}
                        mime = block["source"]["media_type"]
                        data = block["source"]["data"]
                        openai_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{data}"},
                            }
                        )
                messages.append({"role": msg["role"], "content": openai_content})
            else:
                messages.append({"role": msg["role"], "content": content})

    openai_req["messages"] = messages

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
    elif finish_reason == "stop":
        stop_reason = "end_turn"

    # Content
    content = message_data.get("content", "")
    content_blocks = [{"type": "text", "text": content}] if content else []

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
