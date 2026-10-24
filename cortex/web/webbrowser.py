"""
Web Browser — Opens URLs and extracts readable content from web pages.
Used when Cortex needs to actually READ a webpage, not just search for snippets.
"""

import os
import re
import subprocess
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("cortex.webbrowser")

# Headers to mimic a real browser
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags to strip entirely (they add noise, not content)
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe"]


def open_in_browser(url: str) -> dict:
    """
    Open a URL in the user's default browser (cross-platform via osutil).
    Returns: {success, url, message}
    """
    import cortex.util.osutil as osutil
    if osutil.open_url(url):
        logger.info(f"Opened in browser: {url}")
        return {"success": True, "url": url, "message": f"Opened {url} in your browser."}
    return {"success": False, "url": url, "message": f"Couldn't open {url} automatically."}


def fetch_page(url: str, timeout: int = 15) -> dict:
    """
    Fetch a web page and extract readable text content.
    Returns: {success, url, title, content, links, error}
    """
    try:
        response = httpx.get(url, headers=DEFAULT_HEADERS, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # If it's not HTML, return raw text (JSON, plain text, etc.)
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return {
                "success": True,
                "url": str(response.url),
                "title": None,
                "content": response.text[:10000],
                "links": [],
                "content_type": content_type,
            }

        # Parse HTML
        soup = BeautifulSoup(response.text, "lxml")

        # Get title
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        # Remove noise tags
        for tag in soup(STRIP_TAGS):
            tag.decompose()

        # Try to find the main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
            or soup.body
        )

        if not main_content:
            main_content = soup

        # Extract text
        text = _extract_text(main_content)

        # Extract useful links
        links = _extract_links(main_content, base_url=str(response.url))

        logger.info(f"Fetched page: {url} — {len(text)} chars, {len(links)} links")

        return {
            "success": True,
            "url": str(response.url),
            "title": title,
            "content": text[:8000],  # Cap at 8k chars for LLM context
            "links": links[:20],     # Top 20 links
            "content_type": content_type,
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
        return {
            "success": False,
            "url": url,
            "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            "title": None,
            "content": None,
            "links": [],
        }
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "title": None,
            "content": None,
            "links": [],
        }


def fetch_and_format(url: str) -> str:
    """Fetch a page and format it as context for the LLM."""
    result = fetch_page(url)

    if not result["success"]:
        return f"(Failed to fetch {url}: {result.get('error', 'unknown error')})"

    lines = [f"[WEB PAGE: {result.get('title', url)}]"]
    lines.append(f"URL: {result['url']}")
    lines.append("")

    if result.get("content"):
        lines.append(result["content"])

    if result.get("links"):
        lines.append("")
        lines.append("[LINKS ON PAGE]")
        for link in result["links"][:10]:
            lines.append(f"  • {link['text']} → {link['href']}")

    return "\n".join(lines)


def _extract_text(element) -> str:
    """Extract clean readable text from a BeautifulSoup element."""
    # Get all text, preserving some structure
    blocks = []
    for el in element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "pre", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue

        # Add markdown-style headers
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            text = "#" * level + " " + text

        # Add bullet for list items
        if el.name == "li":
            text = "• " + text

        # Code blocks
        if el.name == "pre":
            text = "```\n" + text + "\n```"

        blocks.append(text)

    if blocks:
        return "\n\n".join(blocks)

    # Fallback: just get all text
    return element.get_text(separator="\n", strip=True)


def _extract_links(element, base_url: str) -> list[dict]:
    """Extract meaningful links from a page element."""
    parsed_base = urlparse(base_url)
    links = []
    seen_hrefs = set()

    for a in element.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)

        # Skip empty, anchors, javascript
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if not text or len(text) < 2:
            continue

        # Resolve relative URLs
        if href.startswith("/"):
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        elif not href.startswith("http"):
            continue  # skip mailto:, tel:, etc.

        # Deduplicate
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        links.append({"text": text[:100], "href": href})

    return links
