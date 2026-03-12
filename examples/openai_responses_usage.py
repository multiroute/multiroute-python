from multiroute.openai import OpenAI

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key"


def responses_example():
    """
    Example demonstrating the use of the 'responses' API.
    This also uses the multiroute routing and fallback logic.
    """
    print("\n--- Running Responses API Example ---")
    client = OpenAI()

    try:
        # This will attempt multiroute.ai first
        response = client.responses.create(
            model="gpt-4o", input="Explain how quantum computing works in one sentence."
        )
        print("Response received from API.")
        print(f"Content: {response.output[0].content[0].text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    responses_example()
