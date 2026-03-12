import json
import os
from typing import Any, Dict

import google.genai as genai
import httpx
import openai
from google.genai import types
from google.genai._transformers import t_tools
from google.genai.types import FinishReason, GenerateContentResponseUsageMetadata

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/openai/v1"


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


def _lower_dict_types(d: Any) -> Any:
    """Helper to convert Google's UPPERCASE types to OpenAI's lowercase types."""
    if not isinstance(d, dict):
        return d
    res = {}
    for k, v in d.items():
        if k == "type" and isinstance(v, str):
            res[k] = v.lower()
        elif isinstance(v, dict):
            res[k] = _lower_dict_types(v)
        elif isinstance(v, list):
            res[k] = [
                _lower_dict_types(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            res[k] = v
    return res


def _schema_to_dict(schema: Any) -> Dict[str, Any]:
    if not schema:
        return {}
    d = {}
    if hasattr(schema, "type"):
        d["type"] = schema.type.value if hasattr(schema.type, "value") else schema.type
    if hasattr(schema, "description") and schema.description:
        d["description"] = schema.description
    if hasattr(schema, "properties") and schema.properties:
        d["properties"] = {k: _schema_to_dict(v) for k, v in schema.properties.items()}
    if hasattr(schema, "required") and schema.required:
        d["required"] = schema.required
    if hasattr(schema, "items") and schema.items:
        d["items"] = _schema_to_dict(schema.items)
    return d


def _google_to_openai_request(
    model: str, contents: Any, config: Any = None, client: Any = None
) -> Dict[str, Any]:
    messages = []

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # types.Content object
                role = "user" if item.role == "user" else "assistant"
                content_text = ""
                has_function_call = False
                has_function_response = False
                for part in item.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        name = getattr(fc, "name", "")
                        args = getattr(fc, "args", {})
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": name,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(args),
                                        },
                                    }
                                ],
                            }
                        )
                        has_function_call = True
                    elif hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        name = getattr(fr, "name", "")
                        resp = getattr(fr, "response", {})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": name,
                                "content": json.dumps(resp),
                            }
                        )
                        has_function_response = True
                    elif hasattr(part, "text") and part.text:
                        content_text += part.text
                    elif isinstance(part, dict) and "text" in part:
                        content_text += part["text"]

                if not has_function_call and not has_function_response and content_text:
                    messages.append({"role": role, "content": content_text})
                elif content_text and has_function_call:
                    messages[-1]["content"] = content_text
            elif hasattr(item, "function_call") and item.function_call:
                # Bare Part with function_call
                fc = item.function_call
                name = getattr(fc, "name", "")
                args = getattr(fc, "args", {})
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": name,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                )
            elif hasattr(item, "function_response") and item.function_response:
                # Bare Part with function_response
                fr = item.function_response
                name = getattr(fr, "name", "")
                resp = getattr(fr, "response", {})
                messages.append(
                    {"role": "tool", "tool_call_id": name, "content": json.dumps(resp)}
                )
            elif hasattr(item, "text") and item.text:
                messages.append({"role": "user", "content": item.text})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                role = "user" if role == "user" else "assistant"
                parts = item.get("parts", [])
                content_text = ""
                has_function_response = False
                has_function_call = False
                for p in parts:
                    if isinstance(p, dict):
                        if "functionCall" in p or "function_call" in p:
                            fc = p.get("functionCall") or p.get("function_call")
                            name = (
                                fc.get("name")
                                if isinstance(fc, dict)
                                else getattr(fc, "name", "")
                            )
                            args = (
                                fc.get("args")
                                if isinstance(fc, dict)
                                else getattr(fc, "args", {})
                            )
                            messages.append(
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": name,
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": json.dumps(args),
                                            },
                                        }
                                    ],
                                }
                            )
                            has_function_call = True
                        elif "functionResponse" in p or "function_response" in p:
                            fr = p.get("functionResponse") or p.get("function_response")
                            name = (
                                fr.get("name")
                                if isinstance(fr, dict)
                                else getattr(fr, "name", "")
                            )
                            resp = (
                                fr.get("response")
                                if isinstance(fr, dict)
                                else getattr(fr, "response", {})
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": name,  # OpenAI requires an ID, Google usually uses name
                                    "content": json.dumps(resp),
                                }
                            )
                            has_function_response = True
                        elif "text" in p:
                            content_text += p["text"]
                    elif hasattr(p, "function_call") and p.function_call:
                        fc = p.function_call
                        name = getattr(fc, "name", "")
                        args = getattr(fc, "args", {})
                        messages.append(
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": name,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(args),
                                        },
                                    }
                                ],
                            }
                        )
                        has_function_call = True
                    elif hasattr(p, "function_response") and p.function_response:
                        fr = p.function_response
                        name = getattr(fr, "name", "")
                        resp = getattr(fr, "response", {})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": name,
                                "content": json.dumps(resp),
                            }
                        )
                        has_function_response = True
                    elif hasattr(p, "text") and p.text:
                        content_text += p.text

                if not has_function_response and not has_function_call and content_text:
                    messages.append({"role": role, "content": content_text})
                elif content_text and has_function_call:
                    # Assistant sent text AND tool call
                    messages[-1]["content"] = content_text

    openai_req = {"model": "gemini/" + model, "messages": messages}

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

            if hasattr(config, "tools") and config.tools:
                tools = config.tools
                if client and hasattr(client, "_api_client"):
                    # Use Google's internal transformer if we have a client instance
                    try:
                        g_tools = t_tools(client._api_client, tools)
                        openai_tools = []
                        for t in g_tools:
                            if (
                                hasattr(t, "function_declarations")
                                and t.function_declarations
                            ):
                                for fd in t.function_declarations:
                                    # Convert schema to dict and lowercase the types
                                    params_dict = _schema_to_dict(fd.parameters)
                                    params_dict = _lower_dict_types(params_dict)

                                    openai_tools.append(
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": fd.name,
                                                "description": fd.description or "",
                                                "parameters": params_dict,
                                            },
                                        }
                                    )
                        if openai_tools:
                            openai_req["tools"] = openai_tools
                    except Exception:
                        pass  # Silently drop tools if conversion fails, let backend handle if possible

    return openai_req


def _openai_to_google_response(
    openai_resp: Dict[str, Any], model: str
) -> types.GenerateContentResponse:
    choice = openai_resp.get("choices", [{}])[0]
    message_data = choice.get("message", {})
    content = message_data.get("content", "")

    usage_data = openai_resp.get("usage", {})

    parts = []
    if content:
        parts.append(types.Part(text=content))

    tool_calls = message_data.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            try:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}

            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc["function"]["name"], args=args
                    )
                )
            )

    finish_reason_str = choice.get("finish_reason")
    if finish_reason_str == "stop":
        finish_reason = FinishReason.STOP
    elif finish_reason_str == "length":
        finish_reason = FinishReason.MAX_TOKENS
    elif finish_reason_str == "tool_calls":
        finish_reason = FinishReason.STOP
    else:
        finish_reason = FinishReason.OTHER

    candidate = types.Candidate(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason,
    )

    response = types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=usage_data.get("prompt_tokens", 0),
            candidates_token_count=usage_data.get("completion_tokens", 0),
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
            openai_req = _google_to_openai_request(
                model, contents, config, self._client
            )

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
            openai_req = _google_to_openai_request(
                model, contents, config, self._client
            )

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
