import os
from multiroute.openai import OpenAI, AsyncOpenAI
import asyncio

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key"

def sync_example():
    print("--- Running Sync Example ---")
    client = OpenAI()
    
    # This request will first try https://api.multiroute.ai/v1/chat/completions
    # If that fails (5xx or connection error), it falls back to api.openai.com
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello!"}]
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

async def async_example():
    print("\n--- Running Async Example ---")
    client = AsyncOpenAI()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke."}]
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
