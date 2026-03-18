from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import openai

from multiroute.config import get_multiroute_api_key, get_multiroute_base_url
from multiroute.openai.client import _is_multiroute_error
from multiroute.providers import resolve_model

try:
    from pydantic_ai import Agent as _BaseAgent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, infer_model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
except ImportError as e:
    raise ImportError(
        "pydantic-ai is not installed. "
        "Install it with: pip install 'multiroute[pydantic-ai]'",
    ) from e


def _is_pydantic_ai_multiroute_error(e: Exception) -> bool:
    """Check if an exception is a retryable multiroute error."""
    if isinstance(e, ModelHTTPError) and (
        e.status_code >= 500 or e.status_code in (404, 408)
    ):
        return True
    return bool(_is_multiroute_error(e))


class MultirouteOpenAIProvider(OpenAIProvider):
    """OpenAI provider that routes through the Multiroute proxy first, falling back to native OpenAI."""

    def __init__(
        self,
        multiroute_api_key: str | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=get_multiroute_base_url(),
            api_key=multiroute_api_key,
        )

    @property
    def client(self) -> openai.AsyncOpenAI:
        return self._client

    @asynccontextmanager
    async def _get_client(
        self,
        _settings: ModelSettings | None,
    ) -> AsyncIterator[openai.AsyncOpenAI]:
        yield self._client


class _MultirouteModel(Model):
    """Wraps a pydantic_ai Model to try requests via Multiroute proxy first, then fall back."""

    def __init__(self, original: Model, multiroute_api_key: str | None = None) -> None:
        self._original = original

        proxy = OpenAIChatModel(
            model_name=resolve_model(original.model_name),
            provider=MultirouteOpenAIProvider(multiroute_api_key=multiroute_api_key),
        )
        self._proxy = proxy

    @property
    def model_name(self) -> str:
        return self._original.model_name

    @property
    def system(self) -> str:
        return self._original.system

    async def request(
        self,
        messages: list[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        try:
            return await self._proxy.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except Exception as e:
            if _is_pydantic_ai_multiroute_error(e):
                return await self._original.request(
                    messages,
                    model_settings,
                    model_request_parameters,
                )
            raise

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[Any]:
        try:
            async with self._proxy.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                yield stream
        except Exception as e:
            if _is_pydantic_ai_multiroute_error(e):
                async with self._original.request_stream(
                    messages,
                    model_settings,
                    model_request_parameters,
                    run_context,
                ) as stream:
                    yield stream
            else:
                raise


class Agent(_BaseAgent):  # type: ignore[misc]
    """Drop-in replacement for ``pydantic_ai.Agent`` with automatic Multiroute routing.

    Swap the import — no other changes needed::

        from multiroute.pydantic_ai import Agent

        agent = Agent("openai:gpt-4o")
        result = await agent.run("Hello!")

    When ``MULTIROUTE_API_KEY`` is set, requests are first sent through the
    Multiroute proxy.  On proxy failure the request falls back to the original
    provider transparently.
    """

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        mr_api_key = kwargs.pop("multiroute_api_key", None) or get_multiroute_api_key()

        if not mr_api_key:
            super().__init__(model, **kwargs)
            return

        resolved = infer_model(model) if isinstance(model, str) else model
        super().__init__(
            _MultirouteModel(resolved, multiroute_api_key=mr_api_key),
            **kwargs,
        )
