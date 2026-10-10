"""
Web Search — DuckDuckGo search for when Cortex doesn't know something.
Used as a fallback when the LLM indicates it can't answer from context.
"""

import logging
from ddgs import DDGS

logger = logging.getLogger("cortex.websearch")


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo.
    Returns list of {title, body, href}
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        logger.info(f"Web search '{query[:50]}': {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def search_and_format(query: str, max_results: int = 5) -> str:
    """Search and format results as context for the LLM."""
    results = search(query, max_results)
    if not results:
        return "(No web results found)"

    lines = ["[WEB SEARCH RESULTS]"]
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")[:200]
        href = r.get("href", "")
        lines.append(f"  • {title}")
        lines.append(f"    {body}")
        if href:
            lines.append(f"    Source: {href}")
        lines.append("")

    return "\n".join(lines)
