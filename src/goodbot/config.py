from pydantic_settings import BaseSettings, SettingsConfigDict


class GoodBotConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOODBOT_")

    # LLM model ID (passed to LiteLLM)
    model_id: str = "openai/gpt-4o-mini"

    # Document store paths (directories containing documents to index)
    coding_rules_path: str = "./docs/coding_rules"
    dataset_path: str = "./docs/dataset"
    manuals_path: str = "./docs/manuals"

    # Persisted vector store IDs
    store_ids_path: str = "./.goodbot/store_ids.json"

    # Model parameters
    temperature: float = 0.1
    max_tokens: int = 4096
