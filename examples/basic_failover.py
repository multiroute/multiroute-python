import os
from multiroute.providers.openai import OpenAI

# Initialize the enhanced OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "dummy_key")
)

def main():
    try:
        print("Attempting to get completion with failover...")
        
        # When an invalid model fails, the wrapper will automatically
        # fall back to the next model in the list.
        response = client.chat.completions.create(
            models=["gpt-invalid-model", "gpt-4o-mini", "gpt-3.5-turbo"],
            messages=[
                {"role": "user", "content": "List three features of a routing library."}
            ]
        )
        
        print(f"\nSuccess! Completion retrieved using model: {response.model}")
        print("-" * 40)
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"\nAll models failed. Final error: {e}")

if __name__ == "__main__":
    main()
