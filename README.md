# multiroute-python

A lightweight Python library that wraps popular AI SDKs to provide additional reliability features like automatic failover.

## Overview

`multiroute-python` is designed as a drop-in replacement for the official `openai` SDK. It allows you to specify multiple models for a single request. If the primary model fails (e.g., due to rate limits or API errors), the library automatically retries with the next model in your list.

## Installation

```bash
uv add multiroute
# or
pip install multiroute
```

## Features

- **Automatic Failover**: Specify a list of `models` instead of a single `model`.
- **Drop-in Compatibility**: Works exactly like the `openai` SDK.
- **Async Support**: Full support for `AsyncOpenAI`.

## Usage

### Synchronous Client

```python
from multiroute.providers.openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    # Specify multiple models for automatic failover
    models=["gpt-4", "gpt-3.5-turbo"],
    messages=[{"role": "user", "content": "Explain quantum computing."}]
)

print(f"Used model: {response.model}")
print(response.choices[0].message.content)
```

### Asynchronous Client

```python
import asyncio
from multiroute.providers.openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="your-api-key")

    response = await client.chat.completions.create(
        models=["gpt-4", "gpt-3.5-turbo"],
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Other Providers
`multiroute-python` also supports `anthropic` and `google-genai`. Just replace your standard imports with the `multiroute.providers` equivalents.

**Anthropic:**
```python
from multiroute.providers.anthropic import Anthropic

client = Anthropic(api_key="your-api-key")
response = client.messages.create(
    models=["claude-3-opus", "claude-3-sonnet"],
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Google GenAI:**
```python
from multiroute.providers.google_genai import Client

client = Client(api_key="your-api-key")
response = client.models.generate_content(
    models=["gemini-1.5-pro", "gemini-1.5-flash"],
    contents="Hello!"
)
```

## How It Works

The library subclasses the standard `OpenAI`, `Anthropic`, and `Client` classes and overrides their main generation methods (`chat.completions.create`, `messages.create`, `models.generate_content`, etc.). When you provide a `models` argument:
1. It attempts to call the API with the first model.
2. If it encounters an exception, it catches it and moves to the next model.
3. If all models fail, it raises the last exception encountered.

## Development

To run tests:
```bash
uv run pytest
```

## License

MIT
