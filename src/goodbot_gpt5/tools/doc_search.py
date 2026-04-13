"""Document tools — file reading for coding_rules/dataset, vector search for manuals."""

import os
from pathlib import Path

from langchain_core.tools import tool

from ..config import GoodBotConfig
from ..stores.vector_store import VectorStoreManager

_config = GoodBotConfig()

# Lazy singleton for manuals vector store only
_manager: VectorStoreManager | None = None


def _get_manager() -> VectorStoreManager:
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
    return _manager


_OVERVIEW_FILES = {"DATASET.md", "CODING_RULES.md", "MANUALS.md"}


def _read_overview(directory: str) -> str:
    """Read the directory overview file if one exists."""
    dirpath = Path(directory)
    for name in _OVERVIEW_FILES:
        overview = dirpath / name
        if overview.exists():
            try:
                return overview.read_text(errors="replace")
            except Exception:
                pass
    return ""


def _list_files(directory: str) -> list[str]:
    """List supported files in a directory (non-recursive, skips cases/)."""
    supported = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".py", ".r"}
    dirpath = Path(directory)
    if not dirpath.exists():
        return []
    files = []
    for f in sorted(dirpath.rglob("*")):
        if "cases" in f.parts or ".ipynb_checkpoints" in f.parts:
            continue
        if f.is_file() and f.suffix.lower() in supported:
            files.append(str(f.relative_to(dirpath)))
    return files


_MAX_FILE_SIZE = 50_000  # chars — ~50KB; larger files get truncated


def _read_file(directory: str, filename: str) -> str:
    """Read a file from a directory, returning its contents (truncated if large)."""
    filepath = Path(directory) / filename
    if not filepath.exists():
        available = _list_files(directory)
        return f"File not found: {filename}\n\nAvailable files:\n" + "\n".join(available)
    try:
        content = filepath.read_text(errors="replace")
        if len(content) > _MAX_FILE_SIZE:
            return (
                content[:_MAX_FILE_SIZE]
                + f"\n\n... [TRUNCATED — file is {len(content):,} chars total, showing first {_MAX_FILE_SIZE:,}. "
                f"The file is too large to read in full. Summarize what you see or ask the user for a specific section.]"
            )
        return content
    except Exception as e:
        return f"Error reading {filename}: {e}"


# --- Coding rules tools ---

@tool
def list_coding_rules_files() -> str:
    """List all available coding rules files. Call this first to see what reference files exist before reading specific ones. Returns a directory overview with file descriptions and structure."""
    files = _list_files(_config.coding_rules_path)
    if not files:
        return "No coding rules files found."
    overview = _read_overview(_config.coding_rules_path)
    result = ""
    if overview:
        result += "## Directory Overview\n\n" + overview + "\n\n"
    result += "## Available Files\n\n" + "\n".join(f"- {f}" for f in files)
    return result


@tool
def read_coding_rules_file(filename: str) -> str:
    """Read a specific coding rules file by name. Use list_coding_rules_files first to see available files. These files contain histology codes, site group instructions, and grading dictionaries for cancer registry coding."""
    return _read_file(_config.coding_rules_path, filename)


# --- Dataset tools ---

@tool
def list_dataset_files() -> str:
    """List all available dataset files. Call this first to see what data files exist before reading specific ones. Returns a directory overview with file descriptions and structure."""
    files = _list_files(_config.dataset_path)
    if not files:
        return "No dataset files found."
    overview = _read_overview(_config.dataset_path)
    result = ""
    if overview:
        result += "## Directory Overview\n\n" + overview + "\n\n"
    result += "## Available Files\n\n" + "\n".join(f"- {f}" for f in files)
    return result


@tool
def read_dataset_file(filename: str) -> str:
    """Read a specific dataset file by name. Use list_dataset_files first to see available files. Dataset files include patient demographics, ground truth labels, and clinical notes."""
    return _read_file(_config.dataset_path, filename)


# --- Manuals tool (vector search) ---

@tool
def list_manuals() -> str:
    """List all available reference manuals (PDFs) with descriptions. Call this to see what manuals are available before searching."""
    overview = _read_overview(_config.manuals_path)
    if overview:
        return "## Directory Overview\n\n" + overview
    return "Reference manuals are available at: " + _config.manuals_path


@tool
def search_manuals(query: str) -> str:
    """Search the reference manuals (PDFs) using vector search. Use this when the user asks about SEER staging, FORDS coding rules, solid tumor rules, site/histology validation, or other reference documentation. Use list_manuals first to understand what manuals are available."""
    return _get_manager().search("manuals", query)
