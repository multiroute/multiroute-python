from typing import List, Optional, Any
import anthropic
from anthropic.resources.messages import Messages, AsyncMessages

class MultirouteMessages(Messages):
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

class MultirouteAsyncMessages(AsyncMessages):
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

class Anthropic(anthropic.Anthropic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = MultirouteMessages(self)

class AsyncAnthropic(anthropic.AsyncAnthropic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = MultirouteAsyncMessages(self)
