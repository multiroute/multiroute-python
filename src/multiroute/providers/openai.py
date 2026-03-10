import inspect
from typing import List, Optional, Any

import openai
from openai.resources.chat.completions import Completions, AsyncCompletions
from openai.resources.responses import Responses, AsyncResponses

class MultirouteCompletions(Completions):
    def create(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        
        if models is None:
            return super().create(**kwargs)
            
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
            
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return super().create(**kwargs)
            except Exception as e:
                last_exception = e
                
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")

class MultirouteAsyncCompletions(AsyncCompletions):
    async def create(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        
        if models is None:
            return await super().create(**kwargs)
            
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
            
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return await super().create(**kwargs)
            except Exception as e:
                last_exception = e
                
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")

class MultirouteResponses(Responses):
    def create(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        
        if models is None:
            return super().create(**kwargs)
            
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
            
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return super().create(**kwargs)
            except Exception as e:
                last_exception = e
                
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")

class MultirouteAsyncResponses(AsyncResponses):
    async def create(self, **kwargs) -> Any:
        models: Optional[List[str]] = kwargs.pop('models', None)
        
        if models is None:
            return await super().create(**kwargs)
            
        if not models:
            raise ValueError("Provided `models` list cannot be empty.")
            
        last_exception = None
        for model in models:
            kwargs['model'] = model
            try:
                return await super().create(**kwargs)
            except Exception as e:
                last_exception = e
                
        if last_exception:
            raise last_exception
        raise AssertionError("Unreachable code")


class Chat(openai.resources.chat.Chat):
    def __init__(self, client: openai.OpenAI) -> None:
        super().__init__(client)
        self.completions = MultirouteCompletions(client)

class AsyncChat(openai.resources.chat.AsyncChat):
    def __init__(self, client: openai.AsyncOpenAI) -> None:
        super().__init__(client)
        self.completions = MultirouteAsyncCompletions(client)

class OpenAI(openai.OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = Chat(self)
        self.responses = MultirouteResponses(self)

class AsyncOpenAI(openai.AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = AsyncChat(self)
        self.responses = MultirouteAsyncResponses(self)
