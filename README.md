# multiroute

Official SDK client library for [`multiroute.ai`](https://multiroute.ai).
It provides high-availability wrappers for major LLM providers (OpenAI, Anthropic, Google).

If a `MULTIROUTE_API_KEY` is set, requests are transparently routed through `api.multiroute.ai`.
The SDK adds **client-side failover protection**: if there is any issue talking to `multiroute.ai`
 (network errors, timeouts, 5xx responses, some 404s), the request is automatically retried
 against the original provider, so your call still succeeds whenever the underlying provider is healthy.

### What is this library?

`multiroute` is a thin compatibility layer around the official SDKs:

- **OpenAI**: Drop-in replacement for `openai.OpenAI` / `openai.AsyncOpenAI`.
- **Anthropic**: Drop-in replacement for `anthropic.Anthropic` / `anthropic.AsyncAnthropic`.
- **Google**: Wrapper around `google.genai.Client`.

When `MULTIROUTE_API_KEY` is **not** set, the wrapped clients behave exactly like the original SDKs.
When it **is** set, requests first go through Multiroute; if the proxy returns a retryable error (5xx, timeouts, connection issues, some 404s), the request is automatically retried against the original provider.

### Installation

Install from PyPI. The base install includes the OpenAI client and proxy support:

```bash
pip install multiroute
```

For Anthropic or Google support, install the corresponding extra:

```bash
pip install multiroute[anthropic]   # Anthropic client
pip install multiroute[google]      # Google client
pip install multiroute[all]         # All providers
```

Or with `uv`:

```bash
uv add multiroute
uv add "multiroute[anthropic]"   # optional: Anthropic
uv add "multiroute[google]"      # optional: Google
uv add "multiroute[all]"         # optional: all providers
```

### Configuration

- **Provider API keys**: Configure as you normally would for each provider (for example, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, Google credentials, etc.).
- **Multiroute API key**: Set `MULTIROUTE_API_KEY` in your environment to enable proxy routing and fallback.

Example:

```bash
export OPENAI_API_KEY=sk-...
export MULTIROUTE_API_KEY=sk-...
```

If `MULTIROUTE_API_KEY` is unset, the library is effectively a no-op wrapper around the default clients.

### Usage

#### OpenAI

```python
from multiroute.openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello from multiroute!"}],
)

print(response.choices[0].message.content)
```

Async:

```python
from multiroute.openai import AsyncOpenAI

client = AsyncOpenAI()

async def main() -> None:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello from multiroute (async)!"}],
    )
    print(response.choices[0].message.content)
```

#### Anthropic

```python
from multiroute.anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello from multiroute + Anthropic!"}],
)
```

#### Google

```python
from multiroute.google import Client as GoogleClient

client = GoogleClient()
model = client.models.generate_content

response = model(
    model="gemini-1.5-pro",
    contents="Hello from multiroute + Google!",
)
```

### When to use multiroute

Use `multiroute` if you:

- Want **higher availability** across LLM providers without changing your application code.
- Prefer to keep using the **official SDKs and types**, but add a smart routing / failover layer.
- Need a **simple opt-in** mechanism: setting or unsetting `MULTIROUTE_API_KEY` should be enough.

### Development

If you want to work on the library itself:

- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest`

See `AGENTS.md` for more detailed contributor and agent guidelines.

### License

This project is licensed under the terms of the MIT license. See `LICENSE` for details.
