import asyncio
import json
import os

# Set the providers' keys so litellm can fall back to the actual model if proxy fails:
#   export OPENAI_API_KEY="your-openai-key"
#   export ANTHROPIC_API_KEY="your-anthropic-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing

# Drop-in replacement for `from litellm import completion, acompletion`
from multiroute.litellm import acompletion, completion

# Define a simple tool
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


def get_current_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Mock weather function."""
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": "sunny",
    }


def sync_example():
    print("--- Sync: Basic ---")
    try:
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello!"}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


def sync_streaming_example():
    print("\n--- Sync: Streaming ---")
    try:
        stream = completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Count to three, one number per line."}
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")


async def async_tools_example():
    print("\n--- Async: Tool Use ---")
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]

    try:
        # First turn — model decides to call the tool
        print("First turn: sending request...")
        response = await acompletion(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        if hasattr(response_message, "tool_calls") and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"  Tool call: {function_name}({function_args})")

                if function_name == "get_current_weather":
                    result = get_current_weather(**function_args)
                    print(f"  Tool result: {result}")

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result),
                        }
                    )

            # Second turn — model uses the tool result to form a final answer
            print("Second turn: sending tool result...")
            second_response = await acompletion(
                model="gpt-4o-mini",
                messages=messages,
            )
            print(f"Final response: {second_response.choices[0].message.content}")
        else:
            print(f"Response: {response_message.content}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    sync_example()
    sync_streaming_example()
    asyncio.run(async_tools_example())
