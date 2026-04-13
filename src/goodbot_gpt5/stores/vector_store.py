"""Vector store manager using OpenAI Vector Stores API."""

import json
import logging
import os
from pathlib import Path

from openai import OpenAI

from ..config import GoodBotConfig

logger = logging.getLogger(__name__)

# Supported file extensions for upload
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".py", ".r",
}


class VectorStoreManager:
    """Manages three OpenAI Vector Stores for document search."""

    STORE_NAMES = ["manuals"]

    def __init__(self, config: GoodBotConfig | None = None):
        self._config = config or GoodBotConfig()
        self._client = OpenAI()
        self._store_ids: dict[str, str] = {}
        self._load_store_ids()

    def _store_ids_path(self) -> Path:
        return Path(self._config.store_ids_path)

    def _load_store_ids(self):
        path = self._store_ids_path()
        if path.exists():
            try:
                self._store_ids = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._store_ids = {}

    def _save_store_ids(self):
        path = self._store_ids_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._store_ids, indent=2))

    def _get_doc_path(self, store_name: str) -> str:
        paths = {
            "manuals": self._config.manuals_path,
        }
        return paths[store_name]

    def _validate_store_exists(self, store_id: str) -> bool:
        """Check if a vector store still exists on the API side."""
        try:
            self._client.vector_stores.retrieve(store_id)
            return True
        except Exception:
            return False

    def ensure_store(self, store_name: str) -> str:
        """Ensure a vector store exists and is populated. Returns store ID."""
        if store_name in self._store_ids:
            store_id = self._store_ids[store_name]
            if self._validate_store_exists(store_id):
                return store_id
            # Store was deleted remotely, recreate
            del self._store_ids[store_name]

        store = self._client.vector_stores.create(
            name=f"goodbot_gpt5_{store_name}",
        )
        store_id = store.id
        self._store_ids[store_name] = store_id
        self._save_store_ids()

        # Upload documents from the configured path
        doc_path = self._get_doc_path(store_name)
        self._upload_documents(store_id, doc_path)

        return store_id

    def _upload_documents(self, store_id: str, doc_path: str):
        """Upload all supported files from a directory to a vector store."""
        directory = Path(doc_path)
        if not directory.exists():
            logger.warning(f"Document directory does not exist: {doc_path}")
            return

        for file_path in directory.rglob("*"):
            if "cases" in file_path.parts:
                continue
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    with open(file_path, "rb") as f:
                        self._client.vector_stores.files.upload_and_poll(
                            vector_store_id=store_id,
                            file=f,
                        )
                    logger.info(f"Uploaded {file_path.name} to store {store_id}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")

    def search(self, store_name: str, query: str, k: int = 5) -> str:
        """Search a vector store and return formatted results."""
        store_id = self.ensure_store(store_name)

        try:
            results = self._client.vector_stores.search(
                vector_store_id=store_id,
                query=query,
                max_num_results=k,
            )
        except Exception as e:
            return f"Error searching {store_name}: {e}"

        if not results.data:
            return f"No results found in {store_name} for: {query}"

        formatted = []
        for i, result in enumerate(results.data, 1):
            content = ""
            for content_item in result.content:
                if content_item.type == "text":
                    content += content_item.text
            score = f" (score: {result.score:.3f})" if result.score else ""
            filename = result.filename or "unknown"
            formatted.append(
                f"### Result {i} — {filename}{score}\n{content}"
            )

        return "\n\n".join(formatted)

    def rebuild(self, store_name: str):
        """Delete and recreate a vector store (for when documents change)."""
        if store_name in self._store_ids:
            try:
                self._client.vector_stores.delete(self._store_ids[store_name])
            except Exception:
                pass
            del self._store_ids[store_name]
            self._save_store_ids()

        self.ensure_store(store_name)
