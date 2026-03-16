import asyncio

from multiroute.openai import AsyncOpenAI, OpenAI

# Set your API keys via environment variables before running:
#   export OPENAI_API_KEY="your-openai-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing


def sync_example():
    print("--- Sync: Basic ---")
    client = OpenAI()

    # Tries https://api.multiroute.ai/openai/v1/chat/completions first.
    # Falls back to api.openai.com on 5xx / connection error.
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello!"}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


def sync_streaming_example():
    print("\n--- Sync: Streaming ---")
    client = OpenAI()

    try:
        with client.chat.completions.stream(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count to five, one number per line."},
            ],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()

        # Alternatively, use stream=True directly:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say goodbye!"}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


async def async_example():
    print("\n--- Async: Basic ---")
    client = AsyncOpenAI()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


async def async_streaming_example():
    print("\n--- Async: Streaming ---")
    client = AsyncOpenAI()

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count to three, one number per line."},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    sync_streaming_example()
    asyncio.run(async_example())
    asyncio.run(async_streaming_example())
