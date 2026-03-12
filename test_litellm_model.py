import litellm
import asyncio
import os

litellm._turn_on_debug()

async def main():
    mr_kwargs = {
        "model": "anthropic/claude-3-opus",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": "https://api.multiroute.ai/v1",
        "api_key": "fake",
        "custom_llm_provider": "openai"
    }
    
    try:
        await litellm.acompletion(**mr_kwargs)
    except Exception as e:
        pass

asyncio.run(main())
