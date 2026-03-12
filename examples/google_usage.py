import os
import asyncio
from multiroute.google import Client as GoogleClient

# Set your API keys
# os.environ["MULTIROUTE_API_KEY"] = "your_multiroute_key"
# os.environ["GOOGLE_API_KEY"] = "your_google_key"

def main():
    # Sync usage
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    print("--- Sync Request ---")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say hello in a creative way!"
        )
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage_metadata}")
    except Exception as e:
        print(f"Error: {e}")

async def async_main():
    # Async usage
    client = GoogleClient(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    print("\n--- Async Request ---")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents="What is the capital of France?"
        )
        print(f"Response: {response.text}")
        print(f"Usage: {response.usage_metadata}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    asyncio.run(async_main())
