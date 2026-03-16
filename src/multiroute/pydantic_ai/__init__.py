"""Pydantic AI provider with Multiroute high-availability routing.

Simple usage — just swap the import::

    from multiroute.pydantic_ai import Agent

    agent = Agent("gpt-4o")
    result = await agent.run("Hello!")

Advanced usage with explicit provider::

    from pydantic_ai.models.openai import OpenAIChatModel
    from multiroute.pydantic_ai import Agent, MultirouteOpenAIProvider

    model = OpenAIChatModel(
        'gpt-4o',
        provider=MultirouteOpenAIProvider(api_key='your-openai-api-key'),
    )
    agent = Agent(model)
    result = await agent.run('Hello!')
"""

from multiroute.pydantic_ai.client import Agent, MultirouteOpenAIProvider

__all__ = ["Agent", "MultirouteOpenAIProvider"]
