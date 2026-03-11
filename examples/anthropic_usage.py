import asyncio

from multiroute.anthropic import Anthropic, AsyncAnthropic

# Set your Anthropic API key
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"


def sync_example():
    print("--- Running Sync Example ---")
    client = Anthropic()

    # This request will first try https://api.multiroute.ai/v1/chat/completions
    # If that fails (5xx or connection error), it falls back to api.anthropic.com
    # try:
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    print(f"Response: {response.content[0].text}")
    # except Exception as e:
    #     print(f"Error: {e}")


async def async_example():
    print("\n--- Running Async Example ---")
    client = AsyncAnthropic()

    try:
        response = await client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )
        print(f"Response: {response.content[0].text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
