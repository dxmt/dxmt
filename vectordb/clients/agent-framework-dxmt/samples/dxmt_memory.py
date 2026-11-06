"""DXMT memory as an Agent Framework context provider.

Needs a DXMT server with an embedding model (``ollama pull bge-m3`` on the same machine is
found on its own; ``DXMT_URL`` if the server is not on localhost:9999) and a chat client for
the agent -- this sample uses OpenAI through ``OPENAI_API_KEY``; swap in any ``agent_framework``
chat client.
"""

import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from agent_framework_dxmt import DXMTContextProvider


async def main() -> None:
    # Memories are written under user "alice" and recalled for "alice". A memory written under
    # a shared agent id would only come back if search_agent_id named it.
    async with DXMTContextProvider(user_id="alice", search_user_id="alice") as memory:
        agent = Agent(
            OpenAIChatClient(model_id="gpt-4o-mini"),
            name="assistant",
            instructions="You are a concise assistant.",
            context_providers=[memory],
        )
        await agent.run("I prefer answers in Chinese, and I live in Berlin.")

        # A new run: the preference comes back through recall, by meaning and by keyword.
        response = await agent.run("How should you answer me, and what city am I in?")
        print(response.text)

        # The same memory, directly: remember with supersede, recall, forget.
        kept = await memory.remember("Moved to Munich", kind="fact")
        for m in await memory.recall("where does the user live", limit=3):
            print(f"[{m['score']:.2f}] {m['text']}")
        await memory.forget([kept["id"]])


if __name__ == "__main__":
    asyncio.run(main())
