"""Web search tool using DuckDuckGo HTML lite endpoint."""

import re
from html import unescape
from html.parser import HTMLParser

import requests
from langchain_core.tools import tool

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


class _DuckDuckGoLiteParser(HTMLParser):
    """Parse result links and snippets from DuckDuckGo's HTML lite page."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._current: dict = {}
        self._in_link = False
        self._in_snippet = False
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        # Result links are <a> tags inside the result table with class "result-link"
        if tag == "a" and "result-link" in attrs_dict.get("class", ""):
            self._in_link = True
            self._current = {"href": attrs_dict.get("href", ""), "title": ""}
            self._buf = ""
        # Snippets are in <td> with class "result-snippet"
        if tag == "td" and "result-snippet" in attrs_dict.get("class", ""):
            self._in_snippet = True
            self._buf = ""

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            self._current["title"] = self._buf.strip()
        if tag == "td" and self._in_snippet:
            self._in_snippet = False
            self._current["body"] = self._buf.strip()
            if self._current.get("href"):
                self.results.append(self._current)
            self._current = {}

    def handle_data(self, data):
        if self._in_link or self._in_snippet:
            self._buf += data


def _search_ddg_lite(query: str, max_results: int = 5) -> list[dict]:
    resp = requests.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        headers=_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()

    parser = _DuckDuckGoLiteParser()
    parser.feed(resp.text)
    return parser.results[:max_results]


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this for current information not available in the document stores, such as library documentation, general programming questions, or external references."""
    try:
        results = _search_ddg_lite(query, max_results=5)

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

    except Exception as e:
        return f"Web search error: {e}"


@tool
def web_fetch(url: str) -> str:
    """Fetch the content of a web page and return it as plain text. Use this to download documentation, articles, or other web content found via web_search."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            text = _extract_text_from_html(resp.text)
        else:
            text = resp.text

        # Truncate to avoid blowing up the context
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated]"
        return text

    except Exception as e:
        return f"Web fetch error: {e}"


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags and return readable text."""
    # Remove script and style blocks
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decode entities
    cleaned = unescape(cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()
