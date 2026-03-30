# GoodBot — Jupyter-AI Persona for Clinical Data Extraction

A custom [Jupyter-AI 3.x](https://jupyter-ai.readthedocs.io/) persona (`@goodbot`) built for the **AI for the Public Good** hackathon. GoodBot provides an agentic assistant inside JupyterLab with document search, web search, and notebook manipulation capabilities, designed to help participants extract structured data from clinical notes.

## Features

- **Agentic ReAct loop** — powered by LangGraph with conversation memory
- **Document search** — three configurable vector stores (coding rules, dataset docs, reference manuals) backed by the OpenAI Vector Stores API
- **Web search** — DuckDuckGo integration for external information
- **JupyterLab tools** — read, create, and edit notebook cells; execute code (via Jupyternaut toolkits)
- **Streaming responses** — real-time token streaming in the JupyterLab chat UI
- **Multi-provider LLM support** — uses LiteLLM so any `openai/`-prefixed model works out of the box

## Architecture

```
src/goodbot/
├── persona.py          # GoodBotPersona — entry point, message handling & streaming
├── agent.py            # LangGraph ReAct agent with async SQLite memory
├── chat_models.py      # ChatLiteLLM — LangChain wrapper around LiteLLM
├── config.py           # GoodBotConfig — pydantic-settings with GOODBOT_ env prefix
├── prompt_template.py  # Jinja2 system prompt for clinical data extraction
├── tools/
│   ├── __init__.py     # get_all_tools() — aggregates all tools
│   ├── doc_search.py   # search_coding_rules, search_dataset, search_manuals
│   └── web_search.py   # DuckDuckGo web search
├── stores/
│   └── vector_store.py # VectorStoreManager — OpenAI Vector Stores API
└── static/
    └── goodbot.svg     # Chat avatar
```

## Installation

```bash
# Install in development mode
pip install -e .

# Restart JupyterLab or run /refresh-personas in the chat UI
```

The persona registers itself via the `jupyter_ai.personas` entry point — no manual configuration needed.

## Configuration

All settings use environment variables with the `GOODBOT_` prefix. Every option has a sensible default.

| Variable | Default | Description |
|---|---|---|
| `GOODBOT_MODEL_ID` | `openai/gpt-4o-mini` | LLM model (LiteLLM format) |
| `GOODBOT_TEMPERATURE` | `0.1` | Sampling temperature |
| `GOODBOT_MAX_TOKENS` | `4096` | Max response tokens |
| `GOODBOT_CODING_RULES_PATH` | `./docs/coding_rules` | Path to coding guidelines docs |
| `GOODBOT_DATASET_PATH` | `./docs/dataset` | Path to dataset documentation |
| `GOODBOT_MANUALS_PATH` | `./docs/manuals` | Path to reference manuals |
| `GOODBOT_STORE_IDS_PATH` | `./.goodbot/store_ids.json` | Persisted vector store IDs |

### Document stores

Place files in the configured directories. Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`, `.html`, `.py`, `.r`

Vector store IDs are persisted to `.goodbot/store_ids.json` so documents are only uploaded once. To force a rebuild after changing files, delete the store ID file or call `VectorStoreManager.rebuild(store_name)`.

## Usage

In the JupyterLab chat, mention the persona:

```
@goodbot How do I extract diagnosis codes from the clinical notes?
```

GoodBot will search the relevant document stores, use web search if needed, and can create or edit notebook cells to demonstrate code.

## Requirements

- Python >= 3.11
- JupyterLab with `jupyter_ai >= 3.0.0b7`
- OpenAI API key (for vector stores and embeddings)

See `pyproject.toml` for the full dependency list.
