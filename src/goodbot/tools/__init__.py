from .doc_search import (
    list_coding_rules_files,
    read_coding_rules_file,
    list_dataset_files,
    read_dataset_file,
    list_manuals,
    search_manuals,
)
from .web_search import web_fetch, web_search
from .notebook import (
    list_notebooks,
    read_notebook,
    edit_notebook_cell,
    add_notebook_cell,
    execute_notebook_cell,
)


def get_all_tools():
    """Aggregate all tools available to the GoodBot agent."""
    tools = [
        list_coding_rules_files,
        read_coding_rules_file,
        list_dataset_files,
        read_dataset_file,
        list_manuals,
        search_manuals,
        web_search,
        web_fetch,
        list_notebooks,
        read_notebook,
        edit_notebook_cell,
        add_notebook_cell,
        execute_notebook_cell,
    ]

    return tools
