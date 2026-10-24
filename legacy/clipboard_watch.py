"""
Clipboard Awareness — Watches clipboard for URLs, errors, code snippets.
Runs as a background thread, auto-explains when content changes.
"""

import subprocess
import threading
import time
import re
import json
import urllib.request
import logging

logger = logging.getLogger("cortex.clipboard")

BASE_URL = "http://localhost:7800"


class ClipboardWatcher:
    """Watches clipboard and triggers Cortex analysis on interesting content."""

    def __init__(self, check_interval: float = 2.0):
        self.check_interval = check_interval
        self._last_content = ""
        self._running = False
        self._thread = None
        self._callbacks = []

    def start(self):
        self._running = True
        self._last_content = self._get_clipboard()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="clipboard-watcher")
        self._thread.start()
        logger.info("Clipboard watcher started")

    def stop(self):
        self._running = False

    def on_change(self, callback):
        """Register a callback for clipboard changes: callback(content, analysis)"""
        self._callbacks.append(callback)

    def _watch_loop(self):
        while self._running:
            time.sleep(self.check_interval)
            content = self._get_clipboard()
            if content and content != self._last_content:
                self._last_content = content
                if self._is_interesting(content):
                    analysis = self._analyze(content)
                    for cb in self._callbacks:
                        try:
                            cb(content, analysis)
                        except Exception as e:
                            logger.error(f"Clipboard callback error: {e}")

    def _get_clipboard(self) -> str:
        """Get current clipboard text."""
        try:
            result = subprocess.run(
                ["pbpaste"], capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip()[:2000]
        except Exception:
            return ""

    def _is_interesting(self, content: str) -> bool:
        """Determine if clipboard content is worth analyzing."""
        # URLs
        if re.match(r'https?://', content):
            return True
        # Error messages / stack traces
        if any(kw in content.lower() for kw in ["error", "exception", "traceback", "failed", "fatal"]):
            return True
        # Code snippets (has common code patterns)
        if any(kw in content for kw in ["def ", "function ", "class ", "import ", "const ", "=>", "->"]):
            return True
        # Long enough to be meaningful
        if len(content) > 200:
            return True
        return False

    def _analyze(self, content: str) -> str:
        """Send to Cortex for analysis."""
        try:
            content_type = self._detect_type(content)
            prompt = f"I just copied this to my clipboard ({content_type}). Briefly explain what it is:\n\n{content[:1000]}"

            data = json.dumps({
                "text": prompt,
                "source": "clipboard",
                "session_id": "clipboard"
            }).encode()
            req = urllib.request.Request(
                f"{BASE_URL}/ask",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
            return body.get("speak", "")
        except Exception as e:
            logger.error(f"Clipboard analysis failed: {e}")
            return ""

    def _detect_type(self, content: str) -> str:
        if re.match(r'https?://', content):
            return "URL"
        if "traceback" in content.lower() or "error" in content.lower():
            return "error/stack trace"
        if any(kw in content for kw in ["def ", "function ", "class "]):
            return "code snippet"
        return "text"
