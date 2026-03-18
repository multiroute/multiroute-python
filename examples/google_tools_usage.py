import asyncio
import os

from google.genai import types

from multiroute.google import Client

# Set your API keys via environment variables before running:
#   export GOOGLE_API_KEY="your-google-key"
#   export MULTIROUTE_API_KEY="your-multiroute-key"  # optional — enables proxy routing


# Define a Python function to be used as a tool.
# The Google SDK reads the function signature and docstring to build the tool schema.
def get_weather(location: str) -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g., San Francisco, CA

    """
    # In a real application this would call a weather API.
    # The model only sees the signature and docstring; this body runs locally.
    return f"Sunny, 22°C in {location}"


def sync_tools_example():
    print("--- Sync: Tool Use ---")
    client = Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    contents = ["What is the weather like in Boston?"]

    # First turn — model decides to call the tool
    print("First turn: sending request...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather], temperature=0.0),
    )
    contents.append(response.candidates[0].content)

    if response.function_calls:
        tool_responses = []
        for func_call in response.function_calls:
            print(f"  Tool call: {func_call.name}({func_call.args})")

            # Execute the function locally
            tool_result = {"result": get_weather(**func_call.args)}
            print(f"  Tool result: {tool_result}")

            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=func_call.name,
                        response=tool_result,
                    ),
                ),
            )

        contents.append(types.Content(role="user", parts=tool_responses))

        # Second turn — model uses the tool result to form a final answer
        print("Second turn: sending tool result...")
        final_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        print(f"Final response: {final_response.text}")
    else:
        print(f"Model responded directly: {response.text}")


async def async_tools_example():
    print("\n--- Async: Tool Use ---")
    client = Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    contents = ["What is the weather like in London?"]

    print("First turn: sending request...")
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[get_weather], temperature=0.0),
    )
    contents.append(response.candidates[0].content)

    if response.function_calls:
        tool_responses = []
        for func_call in response.function_calls:
            print(f"  Tool call: {func_call.name}({func_call.args})")

            tool_result = {"result": get_weather(**func_call.args)}
            print(f"  Tool result: {tool_result}")

            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=func_call.name,
                        response=tool_result,
                    ),
                ),
            )

        contents.append(types.Content(role="user", parts=tool_responses))

        print("Second turn: sending tool result...")
        final_response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        print(f"Final response: {final_response.text}")
    else:
        print(f"Model responded directly: {response.text}")


if __name__ == "__main__":
    sync_tools_example()
    asyncio.run(async_tools_example())
