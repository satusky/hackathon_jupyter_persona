import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class GoodBotConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOODBOT_GPT5_")

    # LLM model ID (passed to LiteLLM)
    model_id: str = "openai/gpt-4o-mini"

    # Document store paths (directories containing documents to index)
    coding_rules_path: str = os.path.expanduser("~/coding_rules")
    dataset_path: str = os.path.expanduser("~/dataset")
    manuals_path: str = os.path.expanduser("~/manuals")

    # Persisted vector store IDs
    store_ids_path: str = os.path.expanduser("~/.goodbot_gpt5/store_ids.json")

    # Notebook workspace root (notebook tools resolve paths relative to this)
    notebook_workspace: str = os.path.expanduser("~")

    # Model parameters
    temperature: Optional[float] = 0.1
    max_tokens: int = 4096
