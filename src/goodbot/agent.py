"""Agent factory — mirrors Jupyternaut's agent creation pattern."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Singleton MemorySaver — conversation history persists for the lifetime of the
# JupyterLab server process (but not across restarts).
_memory_store = MemorySaver()


async def build_agent(model, tools, system_prompt, log=None):
    """Build a LangGraph agent with tools, memory, and error handling."""
    return create_react_agent(
        model,
        prompt=system_prompt,
        checkpointer=_memory_store,
        tools=tools,
    )
