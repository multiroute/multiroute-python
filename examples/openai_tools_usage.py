import asyncio
import json

from multiroute.openai import AsyncOpenAI, OpenAI

# Set your API keys via environment variables before running:
#   export OPENAI_API_KEY="your-openai-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing

# Define your tool in standard OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def sync_tools_example():
    print("--- Sync: Tool Use ---")
    client = OpenAI()

    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

    # First turn — model decides to call a tool
    print("First turn: sending request...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    messages.append(message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(
                f"  Tool call: {tool_call.function.name}({tool_call.function.arguments})"
            )

            # Simulate tool execution
            tool_result = "Sunny, 22°C"
            print(f"  Tool result: {tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

        # Second turn — model uses the tool result to form a final answer
        print("Second turn: sending tool result...")
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        print(f"Final response: {final_response.choices[0].message.content}")
    else:
        print(f"Model responded directly: {message.content}")


async def async_tools_example():
    print("\n--- Async: Tool Use ---")
    client = AsyncOpenAI()

    messages = [{"role": "user", "content": "What's the weather like in London today?"}]

    print("First turn: sending request...")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    messages.append(message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"  Tool call: {tool_call.function.name}({args})")

            tool_result = "Cloudy, 14°C"
            print(f"  Tool result: {tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

        print("Second turn: sending tool result...")
        final_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        print(f"Final response: {final_response.choices[0].message.content}")
    else:
        print(f"Model responded directly: {message.content}")


if __name__ == "__main__":
    sync_tools_example()
    asyncio.run(async_tools_example())
