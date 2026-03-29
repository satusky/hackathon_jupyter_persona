"""Web search tool using DuckDuckGo."""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this for current information not available in the document stores, such as library documentation, general programming questions, or external references."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No web results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"### Result {i}: {r.get('title', 'No title')}\n"
                f"{r.get('body', 'No description')}\n"
                f"URL: {r.get('href', 'N/A')}"
            )
        return "\n\n".join(formatted)

    except ImportError:
        return (
            "Web search is unavailable: the `duckduckgo-search` package is not installed. "
            "Install it with `pip install duckduckgo-search`."
        )
    except Exception as e:
        return f"Web search error: {e}"
