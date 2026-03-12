import os
import asyncio
import json

# Set the providers keys so litellm can fall back to the actual model if proxy fails
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
# os.environ["MULTIROUTE_API_KEY"] = "your-multiroute-api-key"

# Instead of `from litellm import completion, acompletion`, use the multiroute wrapper:
from multiroute.litellm import completion, acompletion

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

def get_current_weather(location, unit="fahrenheit"):
    """Mock weather function."""
    return {"location": location, "temperature": "72", "unit": unit, "forecast": "sunny"}

async def run_weather_conversation():
    print("--- Running Weather Tool Conversation ---")
    
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    
    # Step 1: Send the conversation and available tools to the model
    try:
        print(f"Step 1: Asking model about weather...")
        response = await acompletion(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Step 2: Check if the model wanted to call a tool
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            print("Model requested tool calls:")
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f" - Executing: {function_name}({function_args})")
                
                # Execute the mock tool
                if function_name == "get_current_weather":
                    function_response = get_current_weather(
                        location=function_args.get("location"),
                        unit=function_args.get("unit", "fahrenheit")
                    )
                    
                    # Step 3: Send the tool result back to the model
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })
            
            print("Step 2: Sending tool results back to model...")
            second_response = await acompletion(
                model="gpt-4",
                messages=messages
            )
            
            final_content = second_response.choices[0].message.content
            print(f"Final response: {final_content}")
        else:
            print(f"Response: {response_message.content}")
            
    except Exception as e:
        print(f"Error during conversation: {e}")
        print("\nNote: To run this example successfully, ensure MULTIROUTE_API_KEY is set,")
        print("or fallback keys (like OPENAI_API_KEY) are provided.")

if __name__ == "__main__":
    asyncio.run(run_weather_conversation())
