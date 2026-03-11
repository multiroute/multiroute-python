import os
import openai
from openai.resources.chat.completions import (
    Completions as ChatCompletions,
    AsyncCompletions as AsyncChatCompletions,
)
from openai.resources.responses import Responses, AsyncResponses
import httpx
from typing import Any

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/v1"


def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.InternalServerError):  # 5xx errors
        return True
    if isinstance(e, openai.APITimeoutError):
        return True
    if isinstance(
        e, openai.NotFoundError
    ):  # 404 - useful if endpoint or model is missing on proxy
        return True
    return False


class MultirouteChatCompletions(ChatCompletions):
    def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return super().create(**kwargs)

        # Safely create a temporary client sharing the connection pool
        temp_client = self._client.with_options(
            base_url=MULTIROUTE_BASE_URL, api_key=os.environ.get("MULTIROUTE_API_KEY")
        )

        try:
            # Bypass the overridden create and call the original ChatCompletions directly
            return ChatCompletions(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                # Fallback to original
                return super().create(**kwargs)
            raise


class AsyncMultirouteChatCompletions(AsyncChatCompletions):
    async def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await super().create(**kwargs)

        temp_client = self._client.with_options(
            base_url=MULTIROUTE_BASE_URL, api_key=os.environ.get("MULTIROUTE_API_KEY")
        )

        try:
            return await AsyncChatCompletions(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                return await super().create(**kwargs)
            raise


class MultirouteResponses(Responses):
    def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return super().create(**kwargs)

        temp_client = self._client.with_options(
            base_url=MULTIROUTE_BASE_URL, api_key=os.environ.get("MULTIROUTE_API_KEY")
        )

        try:
            return Responses(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                return super().create(**kwargs)
            raise


class AsyncMultirouteResponses(AsyncResponses):
    async def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await super().create(**kwargs)

        temp_client = self._client.with_options(
            base_url=MULTIROUTE_BASE_URL, api_key=os.environ.get("MULTIROUTE_API_KEY")
        )

        try:
            return await AsyncResponses(temp_client).create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                return await super().create(**kwargs)
            raise


class OpenAI(openai.OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the chat completions and responses resources with our wrappers
        self.chat.completions = MultirouteChatCompletions(self)
        self.responses = MultirouteResponses(self)


class AsyncOpenAI(openai.AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat.completions = AsyncMultirouteChatCompletions(self)
        self.responses = AsyncMultirouteResponses(self)
