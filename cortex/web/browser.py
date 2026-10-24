"""
browser.py — real browser automation via Playwright (headless Chromium).

Static scraping (webbrowser.py / webcrawler.py = httpx + BeautifulSoup) can't
see JavaScript-rendered content, and can't click/type/scroll. This module
drives a real Chromium so Cortex can:
  * read fully-rendered pages (React/Vue/SPA sites),
  * log in through JS forms, click, fill, wait, scroll,
  * screenshot and extract structured data.

It is heavier than an httpx GET (a browser launch + a ~150 MB Chromium), so the
agent uses it as a FALLBACK when static fetch comes back empty, or when a task
clearly needs interaction. Playwright is imported lazily so the rest of Cortex
works even before `playwright install chromium` has run.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("cortex.browser")

_DEFAULT_TIMEOUT = 20_000  # ms
_MAX_TEXT = 12_000


def available() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except Exception:
        return False


def _new_page(p, headless=True):
    browser = p.chromium.launch(headless=headless)
    ctx = browser.new_context(
        user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
        viewport={"width": 1280, "height": 900},
    )
    return browser, ctx, ctx.new_page()


def render_page(url: str, wait_selector: str = None, timeout_ms: int = _DEFAULT_TIMEOUT) -> dict:
    """
    Load a URL in a real browser and return the FULLY RENDERED text
    (after JS runs). Use when static fetch returns little/nothing.
    """
    if not available():
        return {"success": False, "error": "playwright_not_installed",
                "hint": "run: .venv/bin/playwright install chromium"}
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return {"success": False, "error": f"playwright_import_failed: {e}"}

    try:
        with sync_playwright() as p:
            browser, ctx, page = _new_page(p)
            try:
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=timeout_ms)
                    except Exception:
                        pass
                else:
                    page.wait_for_timeout(1200)  # let late JS settle
                title = page.title()
                text = page.inner_text("body")
                final_url = page.url
            finally:
                browser.close()
        return {"success": True, "url": final_url, "title": title,
                "text": (text or "")[:_MAX_TEXT], "rendered": True}
    except Exception as e:
        logger.warning(f"render_page failed for {url}: {e}")
        return {"success": False, "error": str(e), "url": url}


def automate(url: str, steps: list[dict], timeout_ms: int = _DEFAULT_TIMEOUT) -> dict:
    """
    Drive a real browser through a sequence of steps and return what was seen.

    Each step is a dict with "action":
      {"action":"fill","selector":"#email","value":"me@x.com"}
      {"action":"click","selector":"button[type=submit]"}
      {"action":"wait","ms":1500}  or  {"action":"wait","selector":".results"}
      {"action":"extract","selector":".price"}   -> collects matching texts
      {"action":"screenshot","path":"/tmp/shot.png"}
    Returns {"success", "url", "title", "text", "extracted": {...}, "log": [...]}.
    """
    if not available():
        return {"success": False, "error": "playwright_not_installed",
                "hint": "run: .venv/bin/playwright install chromium"}
    from playwright.sync_api import sync_playwright

    log, extracted = [], {}
    try:
        with sync_playwright() as p:
            browser, ctx, page = _new_page(p)
            try:
                page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                log.append(f"opened {url}")
                for i, step in enumerate(steps or []):
                    act = step.get("action")
                    try:
                        if act == "fill":
                            page.fill(step["selector"], step.get("value", ""), timeout=timeout_ms)
                            log.append(f"filled {step['selector']}")
                        elif act == "click":
                            page.click(step["selector"], timeout=timeout_ms)
                            log.append(f"clicked {step['selector']}")
                        elif act == "wait":
                            if step.get("selector"):
                                page.wait_for_selector(step["selector"], timeout=timeout_ms)
                            else:
                                page.wait_for_timeout(int(step.get("ms", 1000)))
                            log.append("waited")
                        elif act == "extract":
                            els = page.query_selector_all(step["selector"])
                            vals = [e.inner_text() for e in els][:50]
                            extracted[step["selector"]] = vals
                            log.append(f"extracted {len(vals)} from {step['selector']}")
                        elif act == "screenshot":
                            path = step.get("path", "/tmp/cortex_shot.png")
                            page.screenshot(path=path, full_page=bool(step.get("full_page")))
                            log.append(f"screenshot -> {path}")
                        else:
                            log.append(f"skip unknown step {act}")
                    except Exception as se:
                        log.append(f"step {i} ({act}) failed: {se}")
                title = page.title()
                text = page.inner_text("body")
                final_url = page.url
            finally:
                browser.close()
        return {"success": True, "url": final_url, "title": title,
                "text": (text or "")[:_MAX_TEXT], "extracted": extracted, "log": log}
    except Exception as e:
        logger.warning(f"automate failed for {url}: {e}")
        return {"success": False, "error": str(e), "url": url, "log": log}
