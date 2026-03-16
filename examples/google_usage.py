import asyncio
import os

from multiroute.google import Client as GoogleClient

# Set your API keys via environment variables before running:
#   export GOOGLE_API_KEY="your-google-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing


def sync_example():
    print("--- Sync: Basic ---")
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents="Say hello in a creative way!",
        )
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage_metadata}")
    except Exception as e:
        print(f"Error: {e}")


def sync_streaming_example():
    print("\n--- Sync: Streaming ---")
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        # generate_content_stream routes through the proxy and translates
        # OpenAI SSE chunks back to native GenerateContentResponse objects.
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Count to five, one number per line.",
        ):
            # Each chunk has .text if there is a text part in this chunk
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()  # newline after streamed output
    except Exception as e:
        print(f"Error: {e}")


async def async_example():
    print("\n--- Async: Basic ---")
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash", contents="What is the capital of France?",
        )
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage_metadata}")
    except Exception as e:
        print(f"Error: {e}")


async def async_streaming_example():
    print("\n--- Async: Streaming ---")
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        stream = await client.aio.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Count to three, one number per line.",
        )
        async for chunk in stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    sync_streaming_example()
    asyncio.run(async_example())
    asyncio.run(async_streaming_example())
