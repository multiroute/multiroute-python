from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import openai

from multiroute.config import get_api_key, settings
from multiroute.openai.client import _is_multiroute_error

try:
    from pydantic_ai import Agent as _BaseAgent
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, infer_model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
except ImportError:
    raise ImportError(
        "pydantic-ai is not installed. "
        "Install it with: pip install 'multiroute[pydantic-ai]'"
    )


class MultirouteOpenAIProvider(OpenAIProvider):
    """Pydantic AI OpenAI provider with Multiroute high-availability routing.

    Drop-in replacement for ``pydantic_ai.providers.openai.OpenAIProvider``.
    When a Multiroute API key is available (via the ``multiroute_api_key``
    argument or the ``MULTIROUTE_API_KEY`` environment variable), every
    chat-completions request is first attempted through the Multiroute proxy
    (``https://api.multiroute.ai/openai/v1``).  On proxy failure (5xx,
    connection error, timeout, 404) the request falls back transparently to
    the native OpenAI endpoint.

    Usage::

        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from multiroute.pydantic_ai import MultirouteOpenAIProvider

        model = OpenAIChatModel(
            'gpt-4o',
            provider=MultirouteOpenAIProvider(
                api_key='your-openai-api-key',
                multiroute_api_key='your-multiroute-api-key',
            ),
        )
        agent = Agent(model)
        result = await agent.run('Hello!')
    """

    def __init__(self, multiroute_api_key: str | None = None) -> None:
        mr_api_key = multiroute_api_key or get_api_key()
        if not mr_api_key:
            logging.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to OpenAI "
                "without Multiroute high-availability routing."
            )

        self._client = openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=mr_api_key or "not-set",
        )


class _MultirouteModel(Model):
    """Wraps any pydantic_ai Model with a Multiroute proxy attempt.

    On each request, tries the call via an ``OpenAIChatModel`` backed by
    ``MultirouteOpenAIProvider``.  If the proxy fails with a retryable error
    (5xx, timeout, connection error, 404), falls back to the original model
    exactly as pydantic_ai would have run it without multiroute.
    """

    def __init__(self, original: Model) -> None:
        self._original = original
        self._proxy = OpenAIChatModel(
            original.model_name,
            provider=MultirouteOpenAIProvider(),
        )

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
                messages, model_settings, model_request_parameters
            )
        except Exception as e:
            if _is_multiroute_error(e):
                return await self._original.request(
                    messages, model_settings, model_request_parameters
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
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
        except Exception as e:
            if _is_multiroute_error(e):
                async with self._original.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as stream:
                    yield stream
            else:
                raise


class Agent(_BaseAgent):  # type: ignore[misc]
    """Drop-in replacement for ``pydantic_ai.Agent`` with automatic Multiroute routing.

    Just swap the import — no other changes needed::

        from multiroute.pydantic_ai import Agent

        agent = Agent("gpt-4o")
        result = await agent.run("Hello!")

    When ``MULTIROUTE_API_KEY`` is set, every request is first attempted
    through the Multiroute proxy (``https://api.multiroute.ai/openai/v1``).
    On proxy failure (5xx, timeout, connection error, 404) the request falls
    back transparently to the original provider, exactly as pydantic_ai would
    behave without multiroute.

    All ``pydantic_ai.Agent`` features (tools, streaming, structured output,
    non-OpenAI models, …) work identically.
    """

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        if model is None or not get_api_key():
            # No key — behave exactly like the original Agent.
            super().__init__(model, **kwargs)
            return

        # Resolve any string/KnownModelName to a concrete Model instance using
        # pydantic_ai's own logic, then wrap it for proxy routing.
        resolved = infer_model(model) if isinstance(model, str) else model
        super().__init__(_MultirouteModel(resolved), **kwargs)
