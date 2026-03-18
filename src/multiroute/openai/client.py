import logging
from typing import Any

import openai
from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
from openai.resources.chat.completions import Completions as ChatCompletions
from openai.resources.responses import AsyncResponses, Responses

from multiroute.config import get_multiroute_api_key, get_multiroute_base_url
from multiroute.providers import resolve_model

logger = logging.getLogger(__name__)


def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.InternalServerError):
        return True
    if isinstance(e, openai.APITimeoutError):
        return True
    return bool(isinstance(e, openai.NotFoundError))


class MultirouteChatCompletions(ChatCompletions):
    def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_multiroute_api_key()

        if not mr_api_key:
            return super().create(**kwargs)

        model = kwargs.get("model")
        if model and "/" not in model:
            kwargs["model"] = resolve_model(model, str(self._client.base_url))

        temp_client = self._client.with_options(
            base_url=get_multiroute_base_url(),
            api_key=mr_api_key,
        )

        try:
            return ChatCompletions(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                if model:
                    kwargs["model"] = model
                return super().create(**kwargs)
            raise


class AsyncMultirouteChatCompletions(AsyncChatCompletions):
    async def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_multiroute_api_key()

        if not mr_api_key:
            return await super().create(**kwargs)

        model = kwargs.get("model")
        if model and "/" not in model:
            kwargs["model"] = resolve_model(model, str(self._client.base_url))

        temp_client = self._client.with_options(
            base_url=get_multiroute_base_url(),
            api_key=mr_api_key,
        )

        try:
            return await AsyncChatCompletions(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                # Restore original model if it was changed
                if model:
                    kwargs["model"] = model
                return await super().create(**kwargs)
            raise


class MultirouteResponses(Responses):
    def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_multiroute_api_key()

        if not mr_api_key:
            return super().create(**kwargs)

        model = kwargs.get("model")
        if model and "/" not in model:
            kwargs["model"] = resolve_model(model, str(self._client.base_url))

        temp_client = self._client.with_options(
            base_url=get_multiroute_base_url(),
            api_key=mr_api_key,
        )

        try:
            return Responses(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                if model:
                    kwargs["model"] = model
                return super().create(**kwargs)
            raise


class AsyncMultirouteResponses(AsyncResponses):
    async def create(self, **kwargs) -> Any:
        mr_api_key = kwargs.pop("multiroute_api_key", None)
        if mr_api_key is None:
            mr_api_key = get_multiroute_api_key()

        if not mr_api_key:
            return await super().create(**kwargs)

        model = kwargs.get("model")
        if model and "/" not in model:
            kwargs["model"] = resolve_model(model, str(self._client.base_url))

        temp_client = self._client.with_options(
            base_url=get_multiroute_base_url(),
            api_key=mr_api_key,
        )

        try:
            return await AsyncResponses(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                if model:
                    kwargs["model"] = model
                return await super().create(**kwargs)
            raise


class OpenAI(openai.OpenAI):
    def __init__(self, *args, **kwargs):
        self.multiroute_api_key = (
            kwargs.pop("multiroute_api_key", None) or get_multiroute_api_key()
        )
        super().__init__(*args, **kwargs)
        # Ensure resources are re-assigned after super().__init__
        self.chat.completions = MultirouteChatCompletions(self)
        self.responses = MultirouteResponses(self)
        if not self.multiroute_api_key:
            logger.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to OpenAI "
                "without Multiroute high-availability routing.",
            )


class AsyncOpenAI(openai.AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        self.multiroute_api_key = (
            kwargs.pop("multiroute_api_key", None) or get_multiroute_api_key()
        )
        super().__init__(*args, **kwargs)
        # Ensure resources are re-assigned after super().__init__
        self.chat.completions = AsyncMultirouteChatCompletions(self)
        self.responses = AsyncMultirouteResponses(self)
        if not self.multiroute_api_key:
            logger.error(
                "MULTIROUTE_API_KEY is not set. Requests will go directly to OpenAI "
                "without Multiroute high-availability routing.",
            )
