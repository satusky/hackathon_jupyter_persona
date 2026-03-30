"""Notebook tools — read, edit, create cells, and execute code in .ipynb files."""

import logging
from pathlib import Path

import nbformat
from langchain_core.tools import tool
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

from ..config import GoodBotConfig

_config = GoodBotConfig()
_log = logging.getLogger("goodbot.debug")

# Persistent kernel clients keyed by absolute notebook path
_notebook_clients: dict[str, tuple[KernelManager, NotebookClient]] = {}

OUTPUT_TRUNCATION_LIMIT = 3000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> Path:
    """Resolve path relative to workspace. Validates .ipynb extension and no traversal."""
    workspace = Path(_config.notebook_workspace).resolve()
    target = (workspace / path).resolve()
    if not str(target).startswith(str(workspace)):
        raise ValueError(f"Path traversal not allowed: {path}")
    if target.suffix != ".ipynb":
        raise ValueError(f"Not a notebook file (must end with .ipynb): {path}")
    return target


def _format_outputs(outputs: list) -> str:
    """Format cell outputs into a readable string for the LLM."""
    parts = []
    for out in outputs:
        otype = out.get("output_type", "")
        if otype == "stream":
            name = out.get("name", "stdout")
            text = out.get("text", "")
            parts.append(f"[{name}]: {text}")
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append(f"[result]: {data['text/plain']}")
            elif "image/png" in data or "image/jpeg" in data:
                parts.append("[image output omitted]")
            else:
                parts.append(f"[result]: {list(data.keys())}")
        elif otype == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            tb = "\n".join(out.get("traceback", []))
            parts.append(f"[error]: {ename}: {evalue}\n{tb}")

    result = "\n".join(parts)
    if len(result) > OUTPUT_TRUNCATION_LIMIT:
        result = result[:OUTPUT_TRUNCATION_LIMIT] + "\n... [output truncated]"
    return result


def _get_or_create_client(abs_path: Path) -> tuple[NotebookClient, nbformat.NotebookNode]:
    """Get or create a persistent kernel for the given notebook path."""
    key = str(abs_path)

    if key in _notebook_clients:
        km, client = _notebook_clients[key]
        if km.is_alive():
            # Re-read notebook from disk to pick up any external changes
            nb = nbformat.read(str(abs_path), as_version=4)
            client.nb = nb
            return client, nb

    # Start a new kernel
    km = KernelManager(kernel_name="python3")
    km.start_kernel()
    kc = km.client()
    kc.wait_for_ready(timeout=60)

    nb = nbformat.read(str(abs_path), as_version=4)
    client = NotebookClient(
        nb=nb,
        km=km,
        timeout=120,
        allow_errors=True,
    )
    # Assign the already-started kernel client
    client.kc = kc

    _notebook_clients[key] = (km, client)
    return client, nb


def _save_notebook(nb: nbformat.NotebookNode, path: Path) -> None:
    """Write notebook to disk."""
    nbformat.write(nb, str(path))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def list_notebooks(path: str = "") -> str:
    """List all .ipynb notebook files in the workspace directory. Optionally provide a subdirectory path to narrow the search. Use this to discover available notebooks before reading them."""
    try:
        workspace = Path(_config.notebook_workspace).resolve()
        search_dir = (workspace / path).resolve() if path else workspace
        if not str(search_dir).startswith(str(workspace)):
            return f"Error: path traversal not allowed: {path}"
        if not search_dir.exists():
            return f"Directory not found: {path}"

        notebooks = []
        for f in sorted(search_dir.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in f.parts:
                continue
            notebooks.append(str(f.relative_to(workspace)))

        if not notebooks:
            return f"No notebooks found in {path or 'workspace'}."
        return "Notebooks:\n" + "\n".join(f"- {n}" for n in notebooks)
    except Exception as e:
        return f"Error listing notebooks: {e}"


@tool
def read_notebook(path: str) -> str:
    """Read a notebook and show a summary of all cells with their index, type, source preview, and output summary. Use this to understand notebook structure before editing or executing cells."""
    try:
        abs_path = _resolve_path(path)
        if not abs_path.exists():
            return f"Notebook not found: {path}\nUse list_notebooks() to see available notebooks."

        nb = nbformat.read(str(abs_path), as_version=4)
        if not nb.cells:
            return f"Notebook {path} is empty (no cells)."

        lines = [f"Notebook: {path} ({len(nb.cells)} cells)\n"]
        for i, cell in enumerate(nb.cells):
            source_preview = cell.source[:200]
            if len(cell.source) > 200:
                source_preview += "..."
            source_preview = source_preview.replace("\n", "\\n")

            output_summary = ""
            if cell.cell_type == "code" and cell.get("outputs"):
                output_summary = f" | outputs: {len(cell.outputs)} item(s)"

            lines.append(f"[{i}] {cell.cell_type} | {source_preview}{output_summary}")

        return "\n".join(lines)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading notebook: {e}"


@tool
def edit_notebook_cell(path: str, cell_index: int, new_source: str) -> str:
    """Edit the source code of an existing cell in a notebook. Specify the notebook path, cell index (from read_notebook), and the new source code. Outputs are cleared for code cells."""
    try:
        abs_path = _resolve_path(path)
        if not abs_path.exists():
            return f"Notebook not found: {path}\nUse list_notebooks() to see available notebooks."

        nb = nbformat.read(str(abs_path), as_version=4)
        if cell_index < 0 or cell_index >= len(nb.cells):
            return f"Cell index {cell_index} out of range. Valid range: 0-{len(nb.cells) - 1}"

        cell = nb.cells[cell_index]
        cell.source = new_source
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

        _save_notebook(nb, abs_path)
        return f"Cell [{cell_index}] updated in {path}. If the notebook is open in JupyterLab, click 'Revert' to see changes."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error editing cell: {e}"


@tool
def add_notebook_cell(path: str, source: str, cell_type: str = "code", position: int = -1) -> str:
    """Add a new cell to a notebook. Specify source code, cell_type ('code' or 'markdown'), and position (-1 to append at end). Returns the actual index of the inserted cell."""
    try:
        abs_path = _resolve_path(path)
        if not abs_path.exists():
            return f"Notebook not found: {path}\nUse list_notebooks() to see available notebooks."

        nb = nbformat.read(str(abs_path), as_version=4)

        if cell_type == "code":
            new_cell = nbformat.v4.new_code_cell(source=source)
        elif cell_type == "markdown":
            new_cell = nbformat.v4.new_markdown_cell(source=source)
        else:
            return f"Invalid cell_type: {cell_type}. Use 'code' or 'markdown'."

        if position == -1 or position >= len(nb.cells):
            nb.cells.append(new_cell)
            actual_index = len(nb.cells) - 1
        else:
            pos = max(0, position)
            nb.cells.insert(pos, new_cell)
            actual_index = pos

        _save_notebook(nb, abs_path)
        return f"Added {cell_type} cell at index [{actual_index}] in {path}. If the notebook is open in JupyterLab, click 'Revert' to see changes."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error adding cell: {e}"


@tool
def execute_notebook_cell(path: str, cell_index: int) -> str:
    """Execute a code cell in a notebook and return its output. A persistent kernel is used so variables and imports are preserved across executions. Only code cells can be executed."""
    try:
        abs_path = _resolve_path(path)
        if not abs_path.exists():
            return f"Notebook not found: {path}\nUse list_notebooks() to see available notebooks."

        nb = nbformat.read(str(abs_path), as_version=4)
        if cell_index < 0 or cell_index >= len(nb.cells):
            return f"Cell index {cell_index} out of range. Valid range: 0-{len(nb.cells) - 1}"

        cell = nb.cells[cell_index]
        if cell.cell_type != "code":
            return f"Cell [{cell_index}] is a {cell.cell_type} cell, not a code cell. Only code cells can be executed."

        client, nb = _get_or_create_client(abs_path)

        # Execute the cell
        cell = nb.cells[cell_index]
        client.execute_cell(cell, cell_index)

        # Save updated notebook with outputs
        _save_notebook(nb, abs_path)

        # Format and return outputs
        outputs_text = _format_outputs(cell.outputs)
        if not outputs_text:
            return f"Cell [{cell_index}] executed successfully (no output)."
        return f"Cell [{cell_index}] output:\n{outputs_text}"
    except ValueError as e:
        return f"Error: {e}"
    except TimeoutError:
        return f"Cell [{cell_index}] execution timed out after 120 seconds."
    except RuntimeError as e:
        if "ipykernel" in str(e).lower():
            return f"Kernel error: {e}\nMake sure ipykernel is installed: pip install ipykernel"
        return f"Kernel error: {e}"
    except Exception as e:
        return f"Error executing cell: {e}"


def cleanup_all_kernels() -> None:
    """Shutdown all managed kernels. Called during persona shutdown."""
    for key, (km, client) in list(_notebook_clients.items()):
        try:
            if km.is_alive():
                km.shutdown_kernel(now=True)
                _log.info("Shut down kernel for %s", key)
        except Exception as e:
            _log.warning("Error shutting down kernel for %s: %s", key, e)
    _notebook_clients.clear()
