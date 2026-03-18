"""Official multiroute.ai SDK — use submodules for provider clients.

- OpenAI:  from multiroute.openai import OpenAI, AsyncOpenAI
- Anthropic:  from multiroute.anthropic import Anthropic, AsyncAnthropic
- Google:  from multiroute.google import Client
- LiteLLM:  from multiroute.litellm import completion, acompletion
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
