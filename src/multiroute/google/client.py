import os
from typing import Any, Dict

import google.genai as genai
import httpx
import openai
from google.genai import types

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/v1"


def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(
        e,
        (
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.NotFoundError,
        ),
    ):
        return True
    if isinstance(e, httpx.RequestError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code >= 500 or e.response.status_code == 404:
            return True
    if "google.genai.errors.APIError" in str(type(e)):
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


def _google_to_openai_request(
    model: str, contents: Any, config: Any = None
) -> Dict[str, Any]:
    messages = []

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif hasattr(item, "role") and hasattr(item, "parts"):
                role = "user" if item.role == "user" else "assistant"
                content_parts = []
                for part in item.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, dict) and "text" in part:
                        content_parts.append({"type": "text", "text": part["text"]})
                messages.append(
                    {
                        "role": role,
                        "content": content_parts
                        if len(content_parts) > 1
                        else content_parts[0]["text"]
                        if content_parts
                        else "",
                    }
                )
            elif hasattr(item, "text"):
                messages.append({"role": "user", "content": item.text})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                role = "user" if role == "user" else "assistant"
                parts = item.get("parts", [])
                content_text = ""
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        content_text += p["text"]
                    elif hasattr(p, "text"):
                        content_text += p.text
                messages.append({"role": role, "content": content_text})

    openai_req = {"model": model, "messages": messages}

    if config:
        if isinstance(config, dict):
            if "temperature" in config:
                openai_req["temperature"] = config["temperature"]
            if "top_p" in config:
                openai_req["top_p"] = config["top_p"]
            if "max_output_tokens" in config:
                openai_req["max_tokens"] = config["max_output_tokens"]
            if "stop_sequences" in config:
                openai_req["stop"] = config["stop_sequences"]
        else:
            if hasattr(config, "temperature") and config.temperature is not None:
                openai_req["temperature"] = config.temperature
            if hasattr(config, "top_p") and config.top_p is not None:
                openai_req["top_p"] = config.top_p
            if (
                hasattr(config, "max_output_tokens")
                and config.max_output_tokens is not None
            ):
                openai_req["max_tokens"] = config.max_output_tokens
            if hasattr(config, "stop_sequences") and config.stop_sequences is not None:
                openai_req["stop"] = config.stop_sequences

    return openai_req


def _openai_to_google_response(
    openai_resp: Dict[str, Any], model: str
) -> types.GenerateContentResponse:
    choice = openai_resp.get("choices", [{}])[0]
    message_data = choice.get("message", {})
    content = message_data.get("content", "")

    usage_data = openai_resp.get("usage", {})

    parts = [types.Part(text=content)]
    candidate = types.Candidate(
        content=types.Content(role="model", parts=parts),
        finish_reason="STOP"
        if choice.get("finish_reason") == "stop"
        else "MAX_TOKENS"
        if choice.get("finish_reason") == "length"
        else "OTHER",
    )

    response = types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=types.UsageMetadata(
            prompt_token_count=usage_data.get("prompt_tokens", 0),
            response_token_count=usage_data.get("completion_tokens", 0),
            total_token_count=usage_data.get("total_tokens", 0),
        ),
        model_version=model,
    )
    return response


class MultirouteModels:
    def __init__(self, client: genai.Client, original_method):
        self._client = client
        self._original_generate_content = original_method

    def generate_content(
        self, model: str, contents: Any, config: Any = None, **kwargs
    ) -> types.GenerateContentResponse:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return self._original_generate_content(
                model=model, contents=contents, config=config, **kwargs
            )

        try:
            openai_req = _google_to_openai_request(model, contents, config)

            client = _get_shared_openai_client().with_options(
                api_key=os.environ.get("MULTIROUTE_API_KEY"),
                timeout=kwargs.get("timeout", 60),
            )
            openai_resp_obj = client.chat.completions.create(**openai_req)

            openai_resp = openai_resp_obj.model_dump()
            return _openai_to_google_response(openai_resp, model)
        except Exception as e:
            if _is_multiroute_error(e):
                return self._original_generate_content(
                    model=model, contents=contents, config=config, **kwargs
                )
            raise


class AsyncMultirouteModels:
    def __init__(self, client: genai.Client, original_method):
        self._client = client
        self._original_generate_content = original_method

    async def generate_content(
        self, model: str, contents: Any, config: Any = None, **kwargs
    ) -> types.GenerateContentResponse:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await self._original_generate_content(
                model=model, contents=contents, config=config, **kwargs
            )

        try:
            openai_req = _google_to_openai_request(model, contents, config)

            client = _get_shared_async_openai_client().with_options(
                api_key=os.environ.get("MULTIROUTE_API_KEY"),
                timeout=kwargs.get("timeout", 60),
            )
            openai_resp_obj = await client.chat.completions.create(**openai_req)

            openai_resp = openai_resp_obj.model_dump()
            return _openai_to_google_response(openai_resp, model)
        except Exception as e:
            if _is_multiroute_error(e):
                return await self._original_generate_content(
                    model=model, contents=contents, config=config, **kwargs
                )
            raise


class Client(genai.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Save original methods and override
        self._multiroute_models = MultirouteModels(self, self.models.generate_content)
        self.models.generate_content = self._multiroute_models.generate_content

        if hasattr(self, "aio") and hasattr(self.aio, "models"):
            self._async_multiroute_models = AsyncMultirouteModels(
                self, self.aio.models.generate_content
            )
            self.aio.models.generate_content = (
                self._async_multiroute_models.generate_content
            )
