# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project builds a custom Jupyter-AI 3.x persona (`@goodbot`) for a clinical data extraction hackathon. The persona runs inside a JupyterLab instance in Azure with `jupyter_ai==3.0.0b7`.

## Key Requirements

- **Persona name**: `@goodbot` (referenced in Jupyter-AI chat via `@goodbot`)
- **Capabilities**: Agentic loop (ReAct), document search, web search, JupyterLab MCP tools (cell editing, execution)
- **Three configurable document stores** (OpenAI Vector Stores API):
  - `coding_rules` — coding guidelines
  - `dataset` — dataset documentation
  - `manuals` — reference manuals
- **All LLM calls use the OpenAI SDK** via LiteLLM — models referenced as `openai/` (not `azure/`)
- Default model: `openai/gpt-4o-mini`; embedding: `text-embedding-3-small`
- API keys/endpoints pre-configured as environment variables

## Architecture

```
src/goodbot/
├── persona.py          # GoodBotPersona(BasePersona) — entry point, process_message
├── agent.py            # build_agent() using langchain.agents.create_agent (LangGraph ReAct)
├── chat_models.py      # ChatLiteLLM — LangChain BaseChatModel wrapping LiteLLM
├── config.py           # GoodBotConfig(BaseSettings) — env vars with GOODBOT_ prefix
├── prompt_template.py  # Jinja2 system prompt for clinical data extraction
├── tools/
│   ├── __init__.py     # get_all_tools() — aggregates all tools
│   ├── doc_search.py   # 3 @tool functions using OpenAI Vector Stores
│   └── web_search.py   # DuckDuckGo web search @tool
├── stores/
│   └── vector_store.py # VectorStoreManager — OpenAI Vector Stores API
└── static/
    └── goodbot.svg     # Avatar
```

**Key patterns (from Jupyternaut reference):**
- Persona extends `BasePersona` from `jupyter_ai_persona_manager`
- Agent uses `create_react_agent` from `langgraph.prebuilt` with `MemorySaver` checkpointer
- Streaming via `agent.astream()` with `stream_mode="messages"` → `self.stream_message()`
- Tool errors handled via `wrap_tool_call` middleware
- JupyterLab tools imported from `jupyter_ai_jupyternaut.jupyternaut.toolkits`

## Commands

```bash
# Install in development mode (in target Azure JupyterLab env)
pip install -e .

# After install, restart JupyterLab or use /refresh-personas in chat

# Configuration via environment variables (all optional, have defaults)
export GOODBOT_MODEL_ID="openai/gpt-4o-mini"
export GOODBOT_CODING_RULES_PATH="./docs/coding_rules"
export GOODBOT_DATASET_PATH="./docs/dataset"
export GOODBOT_MANUALS_PATH="./docs/manuals"
```

## Important Notes

- `environment_packages.txt` documents what's installed in the **target** Azure JupyterLab environment — NOT local project dependencies
- The `pyproject.toml` entry point `goodbot = "goodbot.persona:GoodBotPersona"` registers the persona
- Build system is `hatchling` with `packages = ["src/goodbot"]`
- `ChatLiteLLM` in `chat_models.py` is adapted from Jupyternaut — it wraps LiteLLM to work as a LangChain chat model
- Vector store IDs are persisted in `.goodbot/store_ids.json` to avoid re-uploading docs on restart
- `duckduckgo-search` may not be installed in the target env — web search tool handles the ImportError gracefully
