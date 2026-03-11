import os
import openai
from openai.resources.chat.completions import Completions as ChatCompletions, AsyncCompletions as AsyncChatCompletions
from openai.resources.responses import Responses, AsyncResponses
import httpx
from typing import Any

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/v1"

def _is_multiroute_error(e: Exception) -> bool:
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.InternalServerError): # 5xx errors
        return True
    if isinstance(e, openai.APITimeoutError):
        return True
    if isinstance(e, openai.NotFoundError): # 404 - useful if endpoint or model is missing on proxy
        return True
    return False

class MultirouteChatCompletions(ChatCompletions):
    def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return super().create(**kwargs)
        # Save original client's configuration
        original_base_url = self._client.base_url
        original_api_key = self._client.api_key
        try:
            # Temporarily point to multiroute
            self._client.base_url = httpx.URL(MULTIROUTE_BASE_URL)
            self._client.api_key = os.environ.get("MULTIROUTE_API_KEY")
            return super().create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                # Fallback to original
                self._client.base_url = original_base_url
                self._client.api_key = original_api_key
                return super().create(**kwargs)
            raise
        finally:
            # Restore original configuration
            self._client.base_url = original_base_url
            self._client.api_key = original_api_key

class AsyncMultirouteChatCompletions(AsyncChatCompletions):
    async def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await super().create(**kwargs)
        original_base_url = self._client.base_url
        original_api_key = self._client.api_key
        try:
            self._client.base_url = httpx.URL(MULTIROUTE_BASE_URL)
            self._client.api_key = os.environ.get("MULTIROUTE_API_KEY")
            return await super().create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                self._client.base_url = original_base_url
                self._client.api_key = original_api_key
                return await super().create(**kwargs)
            raise
        finally:
            self._client.base_url = original_base_url
            self._client.api_key = original_api_key


class MultirouteResponses(Responses):
    def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return super().create(**kwargs)
        original_base_url = self._client.base_url
        original_api_key = self._client.api_key
        try:
            self._client.base_url = httpx.URL(MULTIROUTE_BASE_URL)
            self._client.api_key = os.environ.get("MULTIROUTE_API_KEY")
            return super().create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                self._client.base_url = original_base_url
                self._client.api_key = original_api_key
                return super().create(**kwargs)
            raise
        finally:
            self._client.base_url = original_base_url
            self._client.api_key = original_api_key

class AsyncMultirouteResponses(AsyncResponses):
    async def create(self, **kwargs) -> Any:
        if not os.environ.get("MULTIROUTE_API_KEY"):
            return await super().create(**kwargs)
        original_base_url = self._client.base_url
        original_api_key = self._client.api_key
        try:
            self._client.base_url = httpx.URL(MULTIROUTE_BASE_URL)
            self._client.api_key = os.environ.get("MULTIROUTE_API_KEY")
            return await super().create(**kwargs)
        except Exception as e:
            if _is_multiroute_error(e):
                self._client.base_url = original_base_url
                self._client.api_key = original_api_key
                return await super().create(**kwargs)
            raise
        finally:
            self._client.base_url = original_base_url
            self._client.api_key = original_api_key


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
