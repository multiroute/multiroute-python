import asyncio

from multiroute.anthropic import Anthropic, AsyncAnthropic

# Set your API keys via environment variables before running:
#   export ANTHROPIC_API_KEY="your-anthropic-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing


def sync_example():
    print("--- Sync: Basic ---")
    client = Anthropic()

    # Tries https://api.multiroute.ai/openai/v1/chat/completions first.
    # Falls back to api.anthropic.com on 5xx / connection error.
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    print(f"Response: {response.content[0].text}")


def sync_streaming_example():
    print("\n--- Sync: Streaming ---")
    client = Anthropic()

    # stream=True routes through the proxy and translates OpenAI SSE chunks
    # back to native Anthropic RawMessageStreamEvent objects.
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Count to five, one word per line."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # newline after streamed output

    # Alternatively, use the raw stream=True parameter:
    stream = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say goodbye!"}],
        stream=True,
    )
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta.text, end="", flush=True)
    print()


async def async_example():
    print("\n--- Async: Basic ---")
    client = AsyncAnthropic()

    try:
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )
        print(f"Response: {response.content[0].text}")
    except Exception as e:
        print(f"Error: {e}")


async def async_streaming_example():
    print("\n--- Async: Streaming ---")
    client = AsyncAnthropic()

    try:
        stream = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Count to three, one word per line."},
            ],
            stream=True,
        )
        async for event in stream:
            if event.type == "content_block_delta":
                print(event.delta.text, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    sync_streaming_example()
    asyncio.run(async_example())
    asyncio.run(async_streaming_example())
