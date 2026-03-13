from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import openai

from multiroute.config import get_api_key, settings
from multiroute.openai.client import _is_multiroute_error

try:
    from pydantic_ai import Agent as _BaseAgent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters, infer_model
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
except ImportError:
    raise ImportError(
        "pydantic-ai is not installed. "
        "Install it with: pip install 'multiroute[pydantic-ai]'",
    )


from multiroute.providers import resolve_model


def _is_pydantic_ai_multiroute_error(e: Exception) -> bool:
    """Check if an exception from pydantic-ai or openai is a multiroute retryable error."""
    # Pydantic AI wraps OpenAI errors into its own ModelHTTPError
    if isinstance(e, ModelHTTPError):
        # 5xx, 404, or 408 (timeout)
        if e.status_code >= 500 or e.status_code in (404, 408):
            return True

    # It might also be a raw OpenAI error if it escaped pydantic-ai's wrapping
    if _is_multiroute_error(e):
        return True

    return False


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

    def __init__(
        self,
        multiroute_api_key: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        mr_api_key = multiroute_api_key or get_api_key()
        if not mr_api_key:
            logging.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to OpenAI "
                "without Multiroute high-availability routing.",
            )

        self._mr_api_key = mr_api_key
        # We must provide a dummy client because OpenAIChatModel expects provider.client to be available
        # immediately during its __init__ (it doesn't wait for _get_client).
        # We use the proxy URL and MR API key here.
        self._client = openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=mr_api_key or "not-set",
        )

    @property
    def client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=self._mr_api_key or "not-set",
        )

    @asynccontextmanager
    async def _get_client(
        self,
        _settings: ModelSettings | None,
    ) -> AsyncIterator[openai.AsyncOpenAI]:
        yield self.client


def _make_proxy_model(
    original: Model,
    multiroute_api_key: str | None = None,
) -> OpenAIChatModel:
    """Create a proxy OpenAIChatModel for the given original model."""
    proxy_model_name = original.model_name
    if "/" not in proxy_model_name:
        # Use the original model's base_url to determine the provider prefix
        # (e.g. "gpt-4o" + "https://api.openai.com/v1/" -> "openai/gpt-4o").
        # If the original already has the multiroute proxy as its base_url (e.g.
        # when the caller passed an OpenAIChatModel(provider=MultirouteOpenAIProvider())),
        # fall back to the default OpenAI base URL so we still get the right prefix.
        original_client = getattr(original, "client", None)
        original_base_url = str(original_client.base_url) if original_client else ""
        if not original_base_url or original_base_url.rstrip(
            "/"
        ) == settings.base_url.rstrip("/"):
            original_base_url = "https://api.openai.com/v1/"
        proxy_model_name = resolve_model(proxy_model_name, original_base_url)

    proxy = OpenAIChatModel(
        proxy_model_name,
        provider=MultirouteOpenAIProvider(multiroute_api_key=multiroute_api_key),
    )
    # HACK: OpenAIChatModel stores the model name in _model_name, which it passes to the client.
    # We must ensure this is the prefixed name so the proxy can route it.
    try:
        object.__setattr__(proxy, "_model_name", proxy_model_name)
    except Exception:
        pass
    return proxy


class _MultirouteModel(Model):
    """Wraps any pydantic_ai Model with a Multiroute proxy attempt.

    On each request, tries the call via an ``OpenAIChatModel`` backed by
    ``MultirouteOpenAIProvider``.  If the proxy fails with a retryable error
    (5xx, timeout, connection error, 404), falls back to the original model
    exactly as pydantic_ai would have run it without multiroute.
    """

    def __init__(self, original: Model, multiroute_api_key: str | None = None) -> None:
        self._multiroute_api_key = multiroute_api_key
        self._proxy = _make_proxy_model(original, multiroute_api_key)
        # The fallback model must be native (not proxy-backed).  If the caller
        # passed an OpenAIChatModel with MultirouteOpenAIProvider, build a plain
        # OpenAIChatModel from the bare model name so fallback goes directly to
        # the upstream provider.
        bare_name = original.model_name
        if "/" in bare_name:
            bare_name = bare_name.split("/", 1)[1]
        self._original: Model = OpenAIChatModel(bare_name)

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
        mr_api_key = kwargs.pop("multiroute_api_key", None) or get_api_key()
        if model is None:
            super().__init__(model, **kwargs)
            return

        # Resolve any string/KnownModelName to a concrete Model instance using
        # pydantic_ai's own logic.
        resolved = infer_model(model) if isinstance(model, str) else model

        if not mr_api_key:
            # No key — bypass the proxy entirely and use a plain OpenAIChatModel
            # so requests go directly to the upstream provider.  We rebuild from
            # the bare model name to avoid carrying over any proxy-backed provider
            # that the caller may have set on the model.
            bare_name = resolved.model_name
            if "/" in bare_name:
                bare_name = bare_name.split("/", 1)[1]
            super().__init__(OpenAIChatModel(bare_name), **kwargs)
            return

        super().__init__(
            _MultirouteModel(resolved, multiroute_api_key=mr_api_key),
            **kwargs,
        )
