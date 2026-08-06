"""
Shell Collector — runs every 1 minute.
Reads the last N commands from zsh_history and stores them in the DB.
Gives Cortex full awareness of what you've been doing in the terminal.
"""

import os
import sys
import re
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context

logger = logging.getLogger("collector.shell")

HISTORY_FILE = os.path.expanduser("~/.zsh_history")
MAX_COMMANDS = 50


def parse_zsh_history(path: str, limit: int = MAX_COMMANDS) -> list[dict]:
    """
    Parse zsh extended history format:
    : timestamp:duration;command
    Falls back to plain format (just the command).
    """
    if not os.path.exists(path):
        return []

    commands = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Cannot read history: {e}")
        return []

    # Read from the end
    for line in reversed(lines):
        line = line.rstrip("\n")
        if not line:
            continue

        # Extended format: ": 1234567890:0;command"
        m = re.match(r"^: (\d+):\d+;(.+)$", line)
        if m:
            ts = int(m.group(1))
            cmd = m.group(2)
            commands.append({
                "command": cmd,
                "timestamp": ts,
                "time": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
            })
        else:
            # Plain format
            if not line.startswith(": "):
                commands.append({
                    "command": line,
                    "timestamp": 0,
                    "time": "",
                })

        if len(commands) >= limit:
            break

    return commands  # Already in reverse-chronological order


def filter_sensitive(commands: list[dict]) -> list[dict]:
    """Strip commands that look like they contain secrets."""
    sensitive_patterns = re.compile(
        r"(password|passwd|secret|token|api.?key|sk-|bearer|auth|credential)",
        re.IGNORECASE
    )
    clean = []
    for c in commands:
        cmd = c["command"]
        if sensitive_patterns.search(cmd):
            # Keep the command but redact the value after = or space
            redacted = re.sub(r'(=\s*|"\s*|\'\s*)([^\s"\']{8,})', r'\1[REDACTED]', cmd)
            c = {**c, "command": redacted}
        clean.append(c)
    return clean


def extract_working_dirs(commands: list[dict]) -> list[str]:
    """Extract unique directories from cd commands."""
    dirs = []
    seen = set()
    for c in commands:
        m = re.match(r"^cd\s+(.+)$", c["command"].strip())
        if m:
            d = m.group(1).strip().strip("'\"")
            d = os.path.expanduser(d)
            if d not in seen:
                seen.add(d)
                dirs.append(d)
    return dirs[:10]


def main():
    config = load_config()
    conn = get_db_conn(config)

    commands = parse_zsh_history(HISTORY_FILE, limit=MAX_COMMANDS)
    commands = filter_sensitive(commands)
    recent_dirs = extract_working_dirs(commands)

    # Most recent command
    latest = commands[0]["command"] if commands else ""

    # Frequency map — what commands are used most
    from collections import Counter
    base_cmds = []
    for c in commands:
        # Take just the base command (first word)
        parts = c["command"].strip().split()
        if parts:
            base_cmds.append(parts[0])
    freq = Counter(base_cmds).most_common(10)

    upsert_context(conn, "shell", "history", {
        "recent_commands": [c["command"] for c in commands[:20]],
        "command_with_times": commands[:20],
        "most_used": [{"cmd": cmd, "count": cnt} for cmd, cnt in freq],
        "recent_dirs": recent_dirs,
        "latest_command": latest,
        "total_captured": len(commands),
    }, expires_minutes=3)

    conn.close()
    logger.info(f"Shell: captured {len(commands)} commands, latest='{latest[:60]}'")


if __name__ == "__main__":
    main()
