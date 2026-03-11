import os
from multiroute.anthropic import Anthropic

# 1. Set your API keys
os.environ["MULTIROUTE_API_KEY"] = os.environ.get(
    "MULTIROUTE_API_KEY", "your-multiroute-key"
)
os.environ["ANTHROPIC_API_KEY"] = os.environ.get(
    "ANTHROPIC_API_KEY", "your-anthropic-key"
)

# 2. Initialize the wrapped client
client = Anthropic()

# 3. Define your tool in standard Anthropic format
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
    }
]

# 4. Create a message with tools (First Turn)
print("Sending request to Multiroute (Anthropic format - First Turn)...")
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=messages,
    tools=tools,
)

# 5. Handle the response and provide tool result (Second Turn)
# Add assistant's response to history
messages.append({"role": "assistant", "content": response.content})

has_tool_use = False
tool_results = []

for block in response.content:
    if block.type == "tool_use":
        has_tool_use = True
        print("\nModel decided to call a tool:")
        print(f" - Tool Name: {block.name}")
        print(f" - Input: {block.input}")

        # Simulate tool execution
        tool_result = "Sunny, 22°C"
        print(f" - Simulated Result: {tool_result}")

        # Construct tool_result block for Anthropic
        tool_results.append(
            {"type": "tool_result", "tool_use_id": block.id, "content": tool_result}
        )
    elif block.type == "text":
        print(f"\nText response: {block.text}")

if has_tool_use:
    # Add tool results to user turn in history
    messages.append({"role": "user", "content": tool_results})

    # Get final response from model
    print("\nSending tool result back to Multiroute (Second Turn)...")
    final_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=messages,
    )
    print(f"\nFinal response: {final_response.content[0].text}")
else:
    print("\nModel responded directly without calling a tool.")
