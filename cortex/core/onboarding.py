"""
onboarding.py — the warm first-run that introduces Cortex and makes it personal.

Cortex is meant to be a "brother in the machine" anyone can use, not just an
engineer. The first time you run `cx` (or `cx setup`), this walks you through a
short, friendly setup: it learns your name and a couple of things you care
about, and explains in plain words what it can do. Everything it learns is
stored locally (owner name in ~/.cortex/owner.json, priorities + a memory fact
in the local SQLite DB).
"""
from __future__ import annotations

import os
import json

_OWNER_FILE = os.path.expanduser("~/.cortex/owner.json")
_DONE_FILE = os.path.expanduser("~/.cortex/.onboarded")


def needs_onboarding() -> bool:
    return not os.path.exists(_DONE_FILE)


def _save_owner(name: str):
    os.makedirs(os.path.dirname(_OWNER_FILE), exist_ok=True)
    with open(_OWNER_FILE, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f)


def mark_done():
    os.makedirs(os.path.dirname(_DONE_FILE), exist_ok=True)
    open(_DONE_FILE, "w", encoding="utf-8").close()


def run_onboarding(console=None) -> None:
    """Interactive first-run. Safe to re-run (idempotent-ish)."""
    if console is None:
        from rich.console import Console
        console = Console()

    console.print()
    console.print("  [bold]Hey — I'm Cortex.[/bold]")
    console.print("  [dim]Think of me as a brother who lives in your machine. I'm on your[/dim]")
    console.print("  [dim]side, I can actually get things done here, and I'll always talk[/dim]")
    console.print("  [dim]to you like a person — no tech-speak unless you want it.[/dim]")
    console.print()

    import sys
    if not sys.stdin.isatty():
        console.print("  [dim](non-interactive — skipping setup; run `cx setup` anytime)[/dim]\n")
        return

    try:
        name = console.input("  What should I call you?  ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n  [dim]No worries — we can do this later with `cx setup`.[/dim]\n")
        return
    if not name:
        name = "friend"
    _save_owner(name)

    console.print(f"\n  Nice to meet you, [bold]{name}[/bold]. A couple quick things so I can be useful.")
    console.print("  [dim]What matters most to you that I should keep an eye on? (one line, or press enter to skip)[/dim]")
    priorities = []
    for i in range(2):
        try:
            p = console.input(f"  • ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not p:
            break
        priorities.append(p)

    # Persist to the local store (best-effort — works offline).
    try:
        from cortex.util.cortex_bootstrap import load_config, connect_db
        db = connect_db(load_config())
        if db:
            db.save_memory(type="fact", content=f"The user's name is {name}.",
                           source="onboarding", importance=5)
            for p in priorities:
                db.create_priority(statement=p, weight=4, source="user")
    except Exception:
        pass

    console.print()
    console.print(f"  [bold]You're all set, {name}.[/bold] Here's what you can just ask me, plainly:")
    console.print("  [dim]•[/dim] \"how's my machine doing?\"  — I'll check and tell you in plain words")
    console.print("  [dim]•[/dim] \"remind me to call mom at 6pm\"")
    console.print("  [dim]•[/dim] \"look up X and summarize it\"  — I can read real web pages")
    console.print("  [dim]•[/dim] \"is my server okay? restart anything that crashed\"  — I'll do it")
    console.print("  [dim]Just talk to me. I'll handle the how.[/dim]")
    console.print()
    mark_done()
