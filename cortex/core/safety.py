"""
Safety gate — shared destructive-action check.
Used by both the live /ask confirmation flow (cortex.py) and the autonomy
loop (autonomy.py) so the two paths can never drift out of sync: nothing
the agent does on its own initiative is held to a looser bar than a live
conversation with the user.
"""

DANGEROUS_PATTERNS = [
    "rm ", "rm -", "kill ", "docker rm", "docker stop",
    "DROP ", "DELETE FROM", "pkill", "shutdown", "reboot",
]

# A much narrower set that must NEVER auto-run, even when Cortex is completing a
# job the user explicitly handed it. These are unrecoverable / whole-system
# destructive. The agent loop (cortex_core) refuses these outright and asks.
import re as _re

_CATASTROPHIC = [
    _re.compile(r"\brm\s+-[a-z]*[rf][a-z]*\s+(-[a-z]+\s+)*(/|~|\$HOME|\.)(\s|$)"),
    _re.compile(r"\bmkfs\b"),
    _re.compile(r"\bdd\b.*\bof=/dev/"),
    _re.compile(r">\s*/dev/sd[a-z]"),
    _re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    _re.compile(r":\(\)\s*\{.*\};\s*:"),            # fork bomb
    _re.compile(r"\bdiskutil\s+(erase|reformat)"),
    _re.compile(r"\bgit\s+push\b.*--force"),
    _re.compile(r"\bchmod\s+-R\s+000\s+/"),
    _re.compile(r"\bsudo\s+rm\b"),
]


def is_catastrophic(command: str) -> tuple[bool, str]:
    """Whole-system-destructive commands that must never auto-run. Returns (bool, why)."""
    cmd = command or ""
    for pat in _CATASTROPHIC:
        if pat.search(cmd):
            return True, f"`{cmd}` looks irreversible/system-destructive — I won't run that on my own."
    return False, ""


def needs_confirmation(response_json: dict) -> tuple[bool, str]:
    """
    Decide whether an agent-proposed action must be held for confirmation
    before it runs. Returns (needs_confirm, confirm_message).
    """
    action = response_json.get("action", "")
    command = response_json.get("command", "") or ""

    if action == "edit" and response_json.get("edit_file"):
        return True, (
            f"I'm going to edit `{response_json['edit_file']}`: "
            f"{response_json.get('edit_instruction', '')}. Go ahead?"
        )

    if action == "remind" and response_json.get("reminder", {}).get("op") == "delete":
        rid = response_json.get("reminder", {}).get("id")
        return True, f"Delete reminder #{rid}?"

    if action in ("execute", "devops") and command:
        if any(p in command for p in DANGEROUS_PATTERNS):
            return True, f"I'm about to run: `{command}`. This looks destructive. Proceed?"

    return False, ""
