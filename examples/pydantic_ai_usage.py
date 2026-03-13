"""Example: Using Pydantic AI with Multiroute high-availability routing.

This example shows how to use MultirouteOpenAIProvider as a drop-in replacement
for pydantic_ai.providers.openai.OpenAIProvider.

Requirements:
    pip install 'multiroute[pydantic-ai]'

Environment variables:
    MULTIROUTE_API_KEY  - Your Multiroute API key (from api.multiroute.ai)
    OPENAI_API_KEY      - Your OpenAI API key (used for fallback)
"""

import asyncio

from pydantic_ai._agent_graph import CallToolsNode, End, ModelRequestNode

from multiroute.pydantic_ai import Agent

# ---------------------------------------------------------------------------
# 1. Simple run
# ---------------------------------------------------------------------------


async def simple_run() -> None:
    """Basic agent.run() call — returns a single result when done."""
    agent = Agent("openai:gpt-4o")

    result = await agent.run("What is the capital of France?")
    print("=== simple_run ===")
    print(result.output)


# ---------------------------------------------------------------------------
# 2. Tool calls
# ---------------------------------------------------------------------------


async def tool_calls() -> None:
    """Register a tool with @agent.tool and let the model call it."""
    agent: Agent[None, str] = Agent(
        "openai:gpt-4o",
        system_prompt="You are a helpful assistant. Use the available tools when needed.",
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Return the current weather for a city (stub implementation)."""
        # In a real application this would call a weather API.
        return f"The weather in {city} is sunny and 22°C."

    result = await agent.run("What is the weather like in Tokyo?")
    print("=== tool_calls ===")
    print(result.output)


# ---------------------------------------------------------------------------
# 3. run_stream — stream text output incrementally
# ---------------------------------------------------------------------------


async def run_stream() -> None:
    """Stream text output from the model as it is generated."""
    agent = Agent("openai:gpt-4o")

    print("=== run_stream ===")
    async with agent.run_stream(
        "Write a two-sentence summary of the Python language."
    ) as stream:
        # stream_text(delta=True) yields each new chunk as it arrives.
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end="", flush=True)
        print()  # newline after streaming finishes

        # The full output is also available after streaming completes.
        full_output = await stream.get_output()
        print(f"[Full output length: {len(full_output)} chars]")


# ---------------------------------------------------------------------------
# 4. agent.iter() — iterate over graph nodes (tool-call events visible)
# ---------------------------------------------------------------------------


async def iter_events() -> None:
    """Use agent.iter() to observe every step the agent takes.

    agent.iter() exposes the internal pydantic-graph nodes:
      - ModelRequestNode  — the agent is about to send a request to the model
      - CallToolsNode     — the model returned tool calls; the agent will execute them
      - End               — the agent has finished and produced a final output
    """
    agent: Agent[None, str] = Agent(
        "openai:gpt-4o",
        system_prompt="Use available tools when needed.",
    )

    @agent.tool_plain
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    print("=== iter_events ===")
    async with agent.iter("What is 123 multiplied by 456?") as agent_run:
        async for node in agent_run:
            if isinstance(node, ModelRequestNode):
                print("[node] ModelRequestNode — sending request to model")
            elif isinstance(node, CallToolsNode):
                print("[node] CallToolsNode — executing tool calls from model")
            elif isinstance(node, End):
                print(f"[node] End — final output: {node.data.output}")

    # The final result is also accessible on agent_run after iteration.
    print(f"Result: {agent_run.result.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await simple_run()
    print()
    await tool_calls()
    print()
    await run_stream()
    print()
    await iter_events()


if __name__ == "__main__":
    asyncio.run(main())
