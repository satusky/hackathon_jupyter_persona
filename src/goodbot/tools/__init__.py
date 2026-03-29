from .doc_search import search_coding_rules, search_dataset, search_manuals
from .web_search import web_search


def get_all_tools():
    """Aggregate all tools available to the GoodBot agent."""
    tools = [
        search_coding_rules,
        search_dataset,
        search_manuals,
        web_search,
    ]

    # Import Jupyternaut toolkits for JupyterLab MCP tools
    try:
        from jupyter_ai_jupyternaut.jupyternaut.toolkits.notebook import (
            toolkit as nb_toolkit,
        )
        from jupyter_ai_jupyternaut.jupyternaut.toolkits.jupyterlab import (
            toolkit as jlab_toolkit,
        )
        from jupyter_ai_jupyternaut.jupyternaut.toolkits.code_execution import (
            toolkit as exec_toolkit,
        )

        tools += nb_toolkit
        tools += jlab_toolkit
        tools += exec_toolkit
    except ImportError:
        pass

    return tools
