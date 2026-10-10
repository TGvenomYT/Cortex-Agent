"""
Credential Prompt — macOS native dialogs for secure credential input.
Uses osascript (AppleScript) to show native input boxes.
The password field is hidden (secure input).

The agent triggers this when it needs credentials it doesn't have saved.
Results are returned to the agent and optionally cached for the session.
"""

import subprocess
import logging
import json
import os
from typing import Optional

logger = logging.getLogger("cortex.credential_prompt")

# Session-level credential cache (lives until daemon restarts)
_session_cache: dict[str, dict] = {}


def prompt_credentials(site_name: str, needs_username: bool = True,
                       needs_password: bool = True, message: str = None) -> dict:
    """
    Show a macOS native dialog to collect credentials.

    Args:
        site_name: Name of the site/service (shown in dialog title)
        needs_username: Whether to ask for username/email
        needs_password: Whether to ask for password/token
        message: Optional custom message in the dialog

    Returns:
        {
            "success": True/False,
            "username": "...",   # or None if not asked
            "password": "...",   # or None if not asked
            "cancelled": False   # True if user hit Cancel
        }
    """
    result = {"success": False, "username": None, "password": None, "cancelled": False}

    # Check session cache first
    cache_key = site_name.lower().strip()
    if cache_key in _session_cache:
        cached = _session_cache[cache_key]
        logger.info(f"Using cached credentials for {site_name}")
        return {"success": True, "username": cached.get("username"), "password": cached.get("password"), "cancelled": False}

    msg = message or f"Cortex needs credentials for {site_name}"

    try:
        # Collect username
        username = None
        if needs_username:
            username = _show_text_dialog(
                title=f"Login — {site_name}",
                message=msg,
                placeholder="Username or email",
                hidden=False
            )
            if username is None:
                result["cancelled"] = True
                return result

        # Collect password
        password = None
        if needs_password:
            pwd_msg = f"Enter password for {username or site_name}"
            password = _show_text_dialog(
                title=f"Password — {site_name}",
                message=pwd_msg,
                placeholder="Password or token",
                hidden=True
            )
            if password is None:
                result["cancelled"] = True
                return result

        result["success"] = True
        result["username"] = username
        result["password"] = password

        # Cache for this session
        _session_cache[cache_key] = {"username": username, "password": password}
        logger.info(f"Credentials collected for {site_name}")

        return result

    except Exception as e:
        logger.error(f"Credential prompt failed: {e}")
        result["success"] = False
        return result


def prompt_single_input(title: str, message: str, placeholder: str = "",
                        hidden: bool = False) -> Optional[str]:
    """
    Show a single input dialog. Returns the entered text or None if cancelled.
    Useful for API keys, tokens, one-off inputs.
    """
    return _show_text_dialog(title=title, message=message, placeholder=placeholder, hidden=hidden)


def prompt_choice(title: str, message: str, choices: list[str]) -> Optional[str]:
    """
    Show a choice dialog with a dropdown. Returns selected item or None if cancelled.
    Useful for: "Which project do you want to run?" type questions.
    """
    choices_str = ", ".join(f'"{c}"' for c in choices)
    script = f'''
    tell application "System Events"
        activate
        set chosenItem to choose from list {{{choices_str}}} ¬
            with title "{_escape(title)}" ¬
            with prompt "{_escape(message)}" ¬
            default items {{"{_escape(choices[0])}"}}
        if chosenItem is false then
            return "CANCELLED"
        else
            return item 1 of chosenItem
        end if
    end tell
    '''
    result = _run_osascript(script)
    if result == "CANCELLED" or result is None:
        return None
    return result


def prompt_confirm(title: str, message: str) -> bool:
    """
    Show a yes/no confirmation dialog. Returns True/False.
    """
    script = f'''
    tell application "System Events"
        activate
        set userChoice to display dialog "{_escape(message)}" ¬
            with title "{_escape(title)}" ¬
            buttons {{"Cancel", "OK"}} ¬
            default button "OK"
        return button returned of userChoice
    end tell
    '''
    result = _run_osascript(script)
    return result == "OK"


def clear_cached_credentials(site_name: str = None):
    """Clear cached credentials. If site_name is None, clear all."""
    if site_name:
        _session_cache.pop(site_name.lower().strip(), None)
    else:
        _session_cache.clear()
    logger.info(f"Cleared credential cache: {site_name or 'all'}")


def get_cached_credential(site_name: str) -> Optional[dict]:
    """Get cached credential if available."""
    return _session_cache.get(site_name.lower().strip())


# ─── Internal helpers ───

def _show_text_dialog(title: str, message: str, placeholder: str = "",
                      hidden: bool = False) -> Optional[str]:
    """Show a text input dialog via osascript. Returns text or None if cancelled."""
    hidden_flag = "with hidden answer" if hidden else ""
    script = f'''
    tell application "System Events"
        activate
        set userInput to display dialog "{_escape(message)}" ¬
            with title "{_escape(title)}" ¬
            default answer "{_escape(placeholder)}" ¬
            buttons {{"Cancel", "OK"}} ¬
            default button "OK" ¬
            {hidden_flag}
        return text returned of userInput
    end tell
    '''
    return _run_osascript(script)


def _run_osascript(script: str) -> Optional[str]:
    """Execute an AppleScript and return the result. Returns None on cancel/error."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes to enter credentials
        )
        if result.returncode != 0:
            # User cancelled or error
            if "User canceled" in result.stderr or result.returncode == 1:
                return None
            logger.warning(f"osascript error: {result.stderr.strip()}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("Credential dialog timed out")
        return None
    except Exception as e:
        logger.error(f"osascript execution failed: {e}")
        return None


def _escape(text: str) -> str:
    """Escape text for AppleScript strings."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
