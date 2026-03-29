"""Document search tools for the three configurable document stores."""

from langchain_core.tools import tool

from ..stores.vector_store import VectorStoreManager

# Lazy singleton — initialized on first use
_manager: VectorStoreManager | None = None


def _get_manager() -> VectorStoreManager:
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
    return _manager


@tool
def search_coding_rules(query: str) -> str:
    """Search the coding rules and guidelines document store. Use this tool when the user asks about coding standards, naming conventions, style guides, or programming rules for the hackathon."""
    return _get_manager().search("coding_rules", query)


@tool
def search_dataset(query: str) -> str:
    """Search the dataset documentation and schemas. Use this tool when the user asks about data fields, table schemas, clinical note formats, or dataset structure."""
    return _get_manager().search("dataset", query)


@tool
def search_manuals(query: str) -> str:
    """Search the reference manuals. Use this tool when the user asks about tools, libraries, processes, or reference documentation."""
    return _get_manager().search("manuals", query)
