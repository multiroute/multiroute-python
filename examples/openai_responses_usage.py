import asyncio

from multiroute.openai import AsyncOpenAI, OpenAI

# Set your API keys via environment variables before running:
#   export OPENAI_API_KEY="your-openai-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing


def sync_responses_example():
    print("--- Sync: Responses API ---")
    client = OpenAI()

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input="Explain how quantum computing works in one sentence.",
        )
        print(f"Response: {response.output[0].content[0].text}")
    except Exception as e:
        print(f"Error: {e}")


def sync_responses_streaming_example():
    print("\n--- Sync: Responses API Streaming ---")
    client = OpenAI()

    try:
        with client.responses.stream(
            model="gpt-4o-mini",
            input="Count to five, one number per line.",
        ) as stream:
            for event in stream:
                if (
                    hasattr(event, "type")
                    and event.type == "response.output_text.delta"
                ):
                    print(event.delta, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


async def async_responses_example():
    print("\n--- Async: Responses API ---")
    client = AsyncOpenAI()

    try:
        response = await client.responses.create(
            model="gpt-4o-mini",
            input="What is the capital of France?",
        )
        print(f"Response: {response.output[0].content[0].text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_responses_example()
    sync_responses_streaming_example()
    asyncio.run(async_responses_example())
