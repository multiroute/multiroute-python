import os
from multiroute.openai import OpenAI

# 1. Set your API keys
# Multiroute will attempt to use MULTIROUTE_API_KEY first.
# If the proxy fails, it will fall back to OPENAI_API_KEY.
os.environ["MULTIROUTE_API_KEY"] = os.environ.get(
    "MULTIROUTE_API_KEY", "your-multiroute-key"
)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your-openai-key")

# 2. Initialize the wrapped client
client = OpenAI()

# 3. Define your tool in standard OpenAI format
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

# 4. Create a chat completion with tools (First Turn)
print("Sending request to Multiroute (OpenAI format - First Turn)...")
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# 5. Handle the response and provide tool result (Second Turn)
message = response.choices[0].message
messages.append(message)  # Add assistant's tool call message to history

if message.tool_calls:
    print(f"\nModel decided to call a tool:")
    for tool_call in message.tool_calls:
        print(f" - Function Name: {tool_call.function.name}")
        print(f" - Arguments: {tool_call.function.arguments}")

        # Simulate tool execution
        tool_result = "Sunny, 22°C"
        print(f" - Simulated Result: {tool_result}")

        # Add tool result to history
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_result,
            }
        )

    # Get final response from model
    print("\nSending tool result back to Multiroute (Second Turn)...")
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    print(f"\nFinal response: {final_response.choices[0].message.content}")
else:
    print(f"\nModel responded directly: {message.content}")
