import asyncio

# Set the providers keys so litellm can fall back to the actual model if proxy fails
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# Instead of `from litellm import completion, acompletion`, use the multiroute wrapper:
from multiroute.litellm import completion, acompletion

# Define a tool in standard OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def sync_example():
    print("--- Running Sync LiteLLM Example ---")

    # This request will first try https://api.multiroute.ai/v1
    # If that fails (5xx or connection error), it falls back natively through litellm
    # to the correct provider (in this case, Anthropic due to the model name).
    try:
        response = completion(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": "What is the weather like in Boston?"}
            ],
            tools=tools,
            tool_choice="auto",
        )
        print(f"Model: {response.model}")

        # The response format matches what users previously expected with litellm
        message = response.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool call returned by model:")
            for tool_call in message.tool_calls:
                print(f" - Function: {tool_call.function.name}")
                print(f" - Arguments: {tool_call.function.arguments}")
        else:
            print(f"Response: {message.content}")

    except Exception as e:
        print(f"Error: {e}")


async def async_example():
    print("\n--- Running Async LiteLLM Example ---")

    try:
        response = await acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=tools,
        )
        print(f"Model: {response.model}")

        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool call returned by model:")
            for tool_call in message.tool_calls:
                print(f" - Function: {tool_call.function.name}")
                print(f" - Arguments: {tool_call.function.arguments}")
        else:
            print(f"Response: {message.content}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
