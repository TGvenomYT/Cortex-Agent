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

    if action == "email_send":
        to = response_json.get("email_to") or "(no recipient)"
        subject = response_json.get("email_subject") or "(no subject)"
        body = (response_json.get("email_body") or "").strip()
        draft = f"To: {to}\nSubject: {subject}" + (f"\n\n{body}" if body else "")
        # Unconditional — there is no "obvious" email that's safe to fire off
        # without the user seeing it first, no matter how clearly they asked.
        return True, f"Here's the draft — nothing has been sent yet:\n\n{draft}\n\nSend it?"

    if action == "workers":
        op, w, prof = response_json.get("workers_op"), response_json.get("worker"), response_json.get("profile")
        # Turning things OFF is always fine. Turning background work ON (it costs battery and,
        # for AI jobs, gigabytes of model memory) is held for an explicit yes — so injected
        # text in a web page can't quietly switch Cortex's autonomy back on.
        if op == "enable":
            return True, f"Turn on the background worker `{w}`? It will keep running on its schedule until you switch it off."
        if op == "profile" and prof in ("balanced", "full"):
            return True, f"Switch background work to the '{prof}' profile? That starts several scheduled jobs" + \
                (", including the AI model." if prof == "full" else ".")

    if action in ("execute", "devops") and command:
        if any(p in command for p in DANGEROUS_PATTERNS):
            return True, f"I'm about to run: `{command}`. This looks destructive. Proceed?"

    return False, ""
