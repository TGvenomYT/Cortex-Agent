"""
Web Crawler — Authenticated web crawling with session management.
Used when Cortex needs to log into a site and navigate through authenticated pages.

Supports:
- Cookie-based session auth (login forms)
- Token/API key auth (Bearer headers)
- Basic auth
- Multi-page crawling with depth control
- Content extraction from authenticated pages
"""

import os
import re
import json
import logging
from urllib.parse import urlparse, urljoin
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("cortex.webcrawler")

# Same browser-like headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

STRIP_TAGS = ["script", "style", "nav", "footer", "aside", "noscript", "svg", "iframe"]


class WebCrawler:
    """
    Authenticated web crawler with session persistence.
    Maintains cookies and auth state across requests.
    """

    def __init__(self, credentials: dict = None, timeout: int = 20):
        """
        Initialize crawler with optional credentials.

        credentials format:
        {
            "auth_type": "form" | "bearer" | "basic" | "cookie" | "header",
            "login_url": "https://site.com/login",      # for form auth
            "username_field": "email",                   # form field name
            "password_field": "password",                # form field name
            "username": "user@example.com",
            "password": "secret",
            "token": "Bearer xyz...",                    # for bearer auth
            "headers": {"X-API-Key": "abc"},            # for custom header auth
            "cookies": {"session_id": "xyz"},           # for pre-set cookies
        }
        """
        self.credentials = credentials or {}
        self.timeout = timeout
        self.authenticated = False
        self.visited = set()
        self.pages = []

        # Create a persistent client with cookie jar
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
            cookies=httpx.Cookies(),
        )

        # Apply auth based on type
        self._apply_auth()

    def _apply_auth(self):
        """Set up authentication on the client."""
        auth_type = self.credentials.get("auth_type", "none")

        if auth_type == "bearer":
            token = self.credentials.get("token", "")
            if not token.startswith("Bearer "):
                token = f"Bearer {token}"
            self._client.headers["Authorization"] = token
            self.authenticated = True
            logger.info("Auth: Bearer token applied")

        elif auth_type == "basic":
            username = self.credentials.get("username", "")
            password = self.credentials.get("password", "")
            self._client.auth = (username, password)
            self.authenticated = True
            logger.info("Auth: Basic auth applied")

        elif auth_type == "header":
            custom_headers = self.credentials.get("headers", {})
            self._client.headers.update(custom_headers)
            self.authenticated = True
            logger.info(f"Auth: Custom headers applied ({len(custom_headers)} headers)")

        elif auth_type == "cookie":
            cookies = self.credentials.get("cookies", {})
            for name, value in cookies.items():
                self._client.cookies.set(name, value)
            self.authenticated = True
            logger.info(f"Auth: Cookies applied ({len(cookies)} cookies)")

        elif auth_type == "form":
            # Form login happens on first request or explicit login() call
            pass

    def login(self) -> dict:
        """
        Perform form-based login.
        Returns: {success, message, cookies_count}
        """
        auth_type = self.credentials.get("auth_type", "")
        if auth_type != "form":
            return {"success": True, "message": "No form login needed for this auth type."}

        login_url = self.credentials.get("login_url")
        if not login_url:
            return {"success": False, "message": "No login_url configured."}

        username_field = self.credentials.get("username_field", "username")
        password_field = self.credentials.get("password_field", "password")
        username = self.credentials.get("username", "")
        password = self.credentials.get("password", "")

        if not username or not password:
            return {"success": False, "message": "Username or password not configured."}

        try:
            # First, GET the login page to pick up CSRF tokens and initial cookies
            login_page = self._client.get(login_url)

            # Look for CSRF token in the form
            csrf_token = None
            extra_fields = {}
            if "text/html" in login_page.headers.get("content-type", ""):
                soup = BeautifulSoup(login_page.text, "html.parser")
                form = soup.find("form")
                if form:
                    # Find hidden inputs (CSRF tokens, etc.)
                    for hidden in form.find_all("input", {"type": "hidden"}):
                        name = hidden.get("name")
                        value = hidden.get("value", "")
                        if name:
                            extra_fields[name] = value
                            if "csrf" in name.lower() or "token" in name.lower():
                                csrf_token = value

            # Build login payload
            payload = {
                username_field: username,
                password_field: password,
                **extra_fields,
            }

            # Also check for any extra_fields in credentials config
            payload.update(self.credentials.get("extra_fields", {}))

            # POST the login form
            response = self._client.post(login_url, data=payload)

            # Check if login succeeded — heuristics:
            # 1. Redirected away from login page
            # 2. Got a session cookie
            # 3. No "invalid" or "error" in response
            success_indicators = [
                response.status_code in (200, 302, 303),
                len(self._client.cookies) > 0,
                str(response.url) != login_url,  # redirected away
            ]

            # Check for failure indicators in response body
            failure_keywords = ["invalid", "incorrect", "failed", "wrong password", "try again"]
            response_lower = response.text[:2000].lower()
            has_failure = any(kw in response_lower for kw in failure_keywords)

            if any(success_indicators) and not has_failure:
                self.authenticated = True
                cookie_count = len(self._client.cookies)
                logger.info(f"Login successful to {login_url} ({cookie_count} cookies)")
                return {
                    "success": True,
                    "message": f"Logged in successfully. {cookie_count} session cookies set.",
                    "cookies_count": cookie_count,
                }
            else:
                logger.warning(f"Login may have failed for {login_url}")
                return {
                    "success": False,
                    "message": "Login appears to have failed. Check credentials.",
                    "response_url": str(response.url),
                }

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return {"success": False, "message": f"Login error: {e}"}

    def fetch(self, url: str) -> dict:
        """
        Fetch a single page using the authenticated session.
        Returns: {success, url, title, content, links, error}
        """
        try:
            response = self._client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # JSON API response
            if "application/json" in content_type:
                try:
                    data = response.json()
                    return {
                        "success": True,
                        "url": str(response.url),
                        "title": None,
                        "content": json.dumps(data, indent=2)[:10000],
                        "links": [],
                        "content_type": content_type,
                        "is_json": True,
                    }
                except Exception:
                    pass

            # Non-HTML text
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
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else None

            for tag in soup(STRIP_TAGS):
                tag.decompose()

            # Find main content
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", {"role": "main"})
                or soup.find("div", class_=re.compile(r"content|article|post|entry|body", re.I))
                or soup.body
                or soup
            )

            text = _extract_text(main_content)
            links = _extract_links(main_content, base_url=str(response.url))

            self.visited.add(url)
            logger.info(f"Crawled: {url} — {len(text)} chars")

            return {
                "success": True,
                "url": str(response.url),
                "title": title,
                "content": text[:8000],
                "links": links[:30],
                "content_type": content_type,
            }

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            # Auth-related failures
            if code in (401, 403):
                msg = f"Access denied ({code}). Credentials may be invalid or session expired."
            else:
                msg = f"HTTP {code}: {e.response.reason_phrase}"
            logger.error(f"Crawl error for {url}: {msg}")
            return {"success": False, "url": url, "error": msg, "title": None, "content": None, "links": []}

        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}")
            return {"success": False, "url": url, "error": str(e), "title": None, "content": None, "links": []}

    def crawl(self, start_url: str, max_pages: int = 5, same_domain: bool = True,
              url_pattern: str = None) -> list[dict]:
        """
        Crawl multiple pages starting from a URL.
        Follows links up to max_pages, optionally filtering by domain or pattern.

        Returns: list of page results [{url, title, content, links}, ...]
        """
        parsed_start = urlparse(start_url)
        base_domain = parsed_start.netloc
        queue = [start_url]
        results = []

        while queue and len(results) < max_pages:
            url = queue.pop(0)

            if url in self.visited:
                continue

            # Domain filter
            if same_domain:
                parsed = urlparse(url)
                if parsed.netloc != base_domain:
                    continue

            # URL pattern filter
            if url_pattern and not re.search(url_pattern, url):
                continue

            page = self.fetch(url)
            if page["success"]:
                results.append(page)
                self.visited.add(url)

                # Queue up links from this page
                for link in page.get("links", []):
                    href = link["href"]
                    if href not in self.visited:
                        queue.append(href)

        logger.info(f"Crawl complete: {len(results)} pages from {start_url}")
        return results

    def close(self):
        """Close the HTTP client and clean up."""
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─── Module-level helpers ───

def crawl_with_credentials(url: str, credentials: dict, max_pages: int = 5,
                           same_domain: bool = True, url_pattern: str = None) -> dict:
    """
    High-level function: create crawler, login if needed, crawl, return results.
    This is what cortex.py calls.

    Returns: {success, pages_crawled, pages: [...], error}
    """
    with WebCrawler(credentials=credentials) as crawler:
        # Login if form-based
        if credentials.get("auth_type") == "form":
            login_result = crawler.login()
            if not login_result["success"]:
                return {
                    "success": False,
                    "pages_crawled": 0,
                    "pages": [],
                    "error": login_result["message"],
                }

        # Crawl
        pages = crawler.crawl(
            start_url=url,
            max_pages=max_pages,
            same_domain=same_domain,
            url_pattern=url_pattern,
        )

        return {
            "success": True,
            "pages_crawled": len(pages),
            "pages": pages,
            "error": None,
        }


def crawl_and_format(url: str, credentials: dict, max_pages: int = 3) -> str:
    """
    Crawl with credentials and format results as LLM context string.
    """
    result = crawl_with_credentials(url, credentials, max_pages=max_pages)

    if not result["success"]:
        return f"(Crawl failed: {result.get('error', 'unknown error')})"

    if not result["pages"]:
        return f"(No pages could be crawled from {url})"

    sections = [f"[CRAWL RESULTS — {result['pages_crawled']} pages from {url}]"]

    for i, page in enumerate(result["pages"], 1):
        title = page.get("title") or page.get("url", "Untitled")
        sections.append(f"\n{'═' * 60}")
        sections.append(f"Page {i}: {title}")
        sections.append(f"URL: {page['url']}")
        sections.append("")
        if page.get("content"):
            sections.append(page["content"][:3000])

    return "\n".join(sections)


# ─── Shared text extraction (same logic as webbrowser.py) ───

def _extract_text(element) -> str:
    """Extract clean readable text from a BeautifulSoup element."""
    blocks = []
    for el in element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "pre", "blockquote"]):
        text = el.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            text = "#" * level + " " + text
        if el.name == "li":
            text = "• " + text
        if el.name == "pre":
            text = "```\n" + text + "\n```"
        blocks.append(text)

    if blocks:
        return "\n\n".join(blocks)

    return element.get_text(separator="\n", strip=True)


def _extract_links(element, base_url: str) -> list[dict]:
    """Extract meaningful links from a page element."""
    parsed_base = urlparse(base_url)
    links = []
    seen = set()

    for a in element.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)

        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        if not text or len(text) < 2:
            continue

        # Resolve relative URLs
        if href.startswith("/"):
            href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        elif not href.startswith("http"):
            href = urljoin(base_url, href)

        if href in seen:
            continue
        seen.add(href)

        links.append({"text": text[:100], "href": href})

    return links
