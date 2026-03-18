import asyncio

from multiroute.anthropic import Anthropic, AsyncAnthropic

# Set your API keys via environment variables before running:
#   export ANTHROPIC_API_KEY="your-anthropic-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing

# Define your tool in standard Anthropic format
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Unit for the temperature",
                },
            },
            "required": ["location"],
        },
    },
]


def sync_tools_example():
    print("--- Sync: Tool Use ---")
    client = Anthropic()

    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

    # First turn — model decides to call a tool
    print("First turn: sending request...")
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )
    messages.append({"role": "assistant", "content": response.content})

    has_tool_use = False
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            has_tool_use = True
            print(f"  Tool call: {block.name}({block.input})")

            # Simulate tool execution
            tool_result = "Sunny, 22°C"
            print(f"  Tool result: {tool_result}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result,
                },
            )
        elif block.type == "text" and block.text:
            print(f"  Text: {block.text}")

    if has_tool_use:
        messages.append({"role": "user", "content": tool_results})

        # Second turn — model uses the tool result to form a final answer
        print("Second turn: sending tool result...")
        final_response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=messages,
        )
        print(f"Final response: {final_response.content[0].text}")
    else:
        print("Model responded directly without calling a tool.")


async def async_tools_example():
    print("\n--- Async: Tool Use ---")
    client = AsyncAnthropic()

    messages = [{"role": "user", "content": "What's the weather like in London?"}]

    print("First turn: sending request...")
    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=messages,
        tools=tools,
    )
    messages.append({"role": "assistant", "content": response.content})

    has_tool_use = False
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            has_tool_use = True
            print(f"  Tool call: {block.name}({block.input})")

            tool_result = "Cloudy, 14°C"
            print(f"  Tool result: {tool_result}")

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result,
                },
            )
        elif block.type == "text" and block.text:
            print(f"  Text: {block.text}")

    if has_tool_use:
        messages.append({"role": "user", "content": tool_results})

        print("Second turn: sending tool result...")
        final_response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=messages,
        )
        print(f"Final response: {final_response.content[0].text}")
    else:
        print("Model responded directly without calling a tool.")


if __name__ == "__main__":
    sync_tools_example()
    asyncio.run(async_tools_example())
