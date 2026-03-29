"""Agent factory — mirrors Jupyternaut's agent creation pattern."""

import os

import aiosqlite
from jupyter_core.paths import jupyter_data_dir
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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


def create_tool_error_handler(log):
    """Creates middleware that catches tool exceptions and returns error messages."""

    @wrap_tool_call
    async def handle_tool_errors(request, handler):
        try:
            return await handler(request)
        except Exception as e:
            log.exception("Tool call raised an exception.")
            return ToolMessage(
                content=f"Tool error: Please check your input and try again. ({str(e)})",
                tool_call_id=request.tool_call["id"],
            )

    return handle_tool_errors


async def build_agent(model, tools, system_prompt, log=None):
    """Build a LangGraph agent with tools, memory, and error handling."""
    memory_store = await get_memory_store()

    middleware = []
    if log:
        middleware.append(create_tool_error_handler(log))

    return create_agent(
        model,
        system_prompt=system_prompt,
        checkpointer=memory_store,
        tools=tools,
        middleware=middleware,
    )
