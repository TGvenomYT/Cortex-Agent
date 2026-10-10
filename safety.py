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
