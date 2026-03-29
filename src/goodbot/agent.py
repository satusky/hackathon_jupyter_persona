"""Agent factory — mirrors Jupyternaut's agent creation pattern."""

import os

import aiosqlite
from jupyter_core.paths import jupyter_data_dir
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent

MEMORY_STORE_PATH = os.path.join(
    jupyter_data_dir(), "jupyter_ai", "goodbot_memory.sqlite"
)


async def get_memory_store(
    _store: dict = {"instance": None},
) -> AsyncSqliteSaver:
    """Get or create the async SQLite memory store (singleton)."""
    if _store["instance"] is None:
        conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
        _store["instance"] = AsyncSqliteSaver(conn)
    return _store["instance"]


async def build_agent(model, tools, system_prompt, log=None):
    """Build a LangGraph agent with tools, memory, and error handling."""
    memory_store = await get_memory_store()

    return create_react_agent(
        model,
        prompt=system_prompt,
        checkpointer=memory_store,
        tools=tools,
    )
