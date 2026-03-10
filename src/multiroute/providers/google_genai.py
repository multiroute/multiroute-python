from typing import List, Optional, Any
from google import genai
from google.genai.models import Models, AsyncModels

class MultirouteModels(Models):
    def generate_content(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        if models is None:
            return super().generate_content(**kwargs)
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return super().generate_content(**kwargs)
            except Exception as e:
                last_exception = e
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")
        
    def generate_content_stream(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        if models is None:
            return super().generate_content_stream(**kwargs)
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return super().generate_content_stream(**kwargs)
            except Exception as e:
                last_exception = e
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")

class MultirouteAsyncModels(AsyncModels):
    async def generate_content(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        if models is None:
            return await super().generate_content(**kwargs)
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return await super().generate_content(**kwargs)
            except Exception as e:
                last_exception = e
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")
        
    async def generate_content_stream(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        if models is None:
            return await super().generate_content_stream(**kwargs)
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return await super().generate_content_stream(**kwargs)
            except Exception as e:
                last_exception = e
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")

import functools
from google.genai.client import AsyncClient

class MultirouteAsyncClient(AsyncClient):
    @functools.cached_property
    def models(self) -> MultirouteAsyncModels:
        return MultirouteAsyncModels(api_client_=self._api_client)

class Client(genai.Client):
    @functools.cached_property
    def models(self) -> MultirouteModels:
        return MultirouteModels(api_client_=self._api_client)

    @functools.cached_property
    def aio(self) -> MultirouteAsyncClient:
        return MultirouteAsyncClient(api_client=self._api_client)
