import os
from multiroute.google import Client
from google.genai import types

# 1. Set your API keys
os.environ["MULTIROUTE_API_KEY"] = os.environ.get(
    "MULTIROUTE_API_KEY", "your-multiroute-key"
)
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "your-google-key")

# 2. Initialize the wrapped client
client = Client()


# 3. Define a Python function to be used as a tool
def get_weather(location: str) -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g., San Francisco, CA
    """
    # This function body is what we'd execute if we were building the full loop.
    # The model only sees the signature and docstring.
    pass


# 4. Generate content with the tool (First Turn)
print("Sending request to Multiroute (Google format - First Turn)...")
contents = ["What is the weather like in Boston?"]
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=types.GenerateContentConfig(tools=[get_weather], temperature=0.0),
)

# 5. Handle the response and provide tool result (Second Turn)
# Add assistant's tool call message to history
contents.append(response.candidates[0].content)

if response.function_calls:
    print("\nModel decided to call a tool:")
    tool_responses = []
    for func_call in response.function_calls:
        print(f" - Function Name: {func_call.name}")
        print(f" - Arguments: {func_call.args}")

        # Simulate tool execution
        tool_result = {"result": "Sunny, 22°C"}
        print(f" - Simulated Result: {tool_result}")

        # Add tool response to current turn
        tool_responses.append(
            types.Part(
                function_response=types.FunctionResponse(
                    name=func_call.name, response=tool_result
                )
            )
        )

    # Add new user message with tool results to history
    contents.append(types.Content(role="user", parts=tool_responses))

    # Get final response from model
    print("\nSending tool result back to Multiroute (Second Turn)...")
    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )
    print(f"\nFinal response: {final_response.text}")
else:
    print(f"\nModel responded directly: {response.text}")
