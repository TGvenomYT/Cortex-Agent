#!/usr/bin/env python3
"""
cortex_cli.py — JSON bridge so other agents (agentkit's MCP server, and thus
Claude Code) can drive Cortex WITHOUT the retired :7800 server.

The MCP server runs under system python; Cortex needs its own venv (psycopg2,
httpx, playwright, …). So agentkit shells out to THIS script with the Cortex
venv, which runs Cortex in-process against local SQLite (works offline).

Usage (stdout is a single JSON object):
  python3 cortex_cli.py ask           '{"text":"..."}'
  python3 cortex_cli.py remember      '{"content":"...","type":"fact","importance":4}'
  python3 cortex_cli.py memory_search '{"query":"...","limit":50}'
  python3 cortex_cli.py reminders     '{}'
  python3 cortex_cli.py goals         '{}'
  python3 cortex_cli.py briefing      '{}'
"""
import os
import sys
import json

os.chdir(__import__("cortex").ROOT)


def _db():
    from cortex.util.cortex_bootstrap import load_config, connect_db
    return connect_db(load_config())


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"bad json args: {e}"}))
        return 2

    if cmd == "ask":
        from cortex.core.cortex_core import CortexCore
        r = CortexCore().ask(args.get("text", ""),
                             session_id=args.get("session_id", "mcp"), source="agentkit")
        print(json.dumps({"speak": r.get("speak", ""), "steps": r.get("steps", 0)}))

    elif cmd == "briefing":
        from cortex.core.cortex_core import CortexCore
        r = CortexCore().ask(
            "Give me a quick, plain-language status update: how's the machine and "
            "my servers, anything that needs my attention, and anything on my plate.",
            session_id="mcp-briefing", source="agentkit")
        print(json.dumps({"speak": r.get("speak", "")}))

    elif cmd == "remember":
        db = _db()
        if db is None:
            print(json.dumps({"error": "db_unavailable"}))
            return 3
        db.save_memory(type=args.get("type", "fact"), content=args["content"],
                       source="agentkit", importance=args.get("importance", 4))
        print(json.dumps({"ok": True}))

    elif cmd == "memory_search":
        db = _db()
        mems = db.get_memories(limit=args.get("limit", 50)) if db else []
        q = (args.get("query") or "").lower().strip()
        if q:
            mems = [m for m in mems if q in str(m).lower()]
        print(json.dumps({"memories": mems, "count": len(mems)}, default=str))

    elif cmd == "reminders":
        db = _db()
        rem = db.get_all_reminders(include_fired=False) if db else []
        print(json.dumps({"reminders": [dict(r) for r in rem], "count": len(rem)}, default=str))

    elif cmd == "goals":
        db = _db()
        goals = db.get_active_goals() if db else []
        print(json.dumps({"goals": [dict(g) for g in goals], "count": len(goals)}, default=str))

    else:
        print(json.dumps({"error": "unknown command", "command": cmd}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
