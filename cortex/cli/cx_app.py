#!/usr/bin/env python3
"""
cx — Cortex Interactive Shell
Talk to your machine. Type naturally, get smart responses.

Usage:
  cx              → interactive session
  cx <one-liner>  → quick one-shot query
"""

import sys
import os
import json
import signal
import threading
import time
try:
    import readline  # nicer line editing on macOS/Linux; absent on Windows
except ImportError:
    readline = None
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown

SESSION_ID = f"cli-{datetime.now().strftime('%Y%m%d-%H%M')}"

# Force a reasonable width even when piped
console = Console(width=100 if not sys.stdout.isatty() else None)

_CORE = None  # in-process Cortex, created on first use


def core():
    global _CORE
    if _CORE is None:
        from cortex.core.cortex_core import CortexCore
        _CORE = CortexCore()
    return _CORE


def follow(events) -> dict:
    """Show a job as it happens. -> its final action."""
    status = console.status("[dim]thinking…[/dim]", spinner="dots") if console.is_terminal else None
    if status:
        status.start()
    written, final, undoable = "", {}, []

    def stop(*_):       # Ctrl+C stops the job, not the app
        console.print("\n  [yellow]stopping…[/yellow]")
        core().stop(SESSION_ID)

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, stop)

    def pause_spinner():
        nonlocal status
        if status:
            status.stop()
            status = None

    try:
        for ev in events:
            t = ev.get("type")
            if t == "delta":
                pause_spinner()
                written += ev["text"]
                print(ev["text"], end="", flush=True)
            elif t == "tool":
                if written:             # the model talking on its way to a step
                    print()
                    written = ""
                pause_spinner()
                detail = f" · {ev['detail']}" if ev.get("detail") else ""
                console.print(f"  [dim]{ev.get('name')}{detail}[/dim]", highlight=False)
            elif t in ("note", "subagent") and ev.get("text"):
                pause_spinner()
                console.print(f"  [dim]{ev['text']}[/dim]", highlight=False)
            elif t == "image":
                pause_spinner()
                console.print(f"  [dim]chart saved in ~/.cortex/artifacts{ev.get('src', '').replace('/artifact', '')}[/dim]")
            elif t == "undoable":
                undoable.append(ev)
            elif t == "final":
                final = ev.get("action") or {}
    finally:
        pause_spinner()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    speak = final.get("speak") or ""
    if written and speak.startswith(written.strip()):
        rest = speak[len(written.strip()):].strip()    # e.g. a heads-up added after the reply
        print("\n")
        if rest:
            console.print(Markdown(rest))
            console.print()
    elif speak:
        print()
        console.print(Markdown(speak))
        console.print()
    if undoable:
        console.print("  [dim]`cx undo` puts those changes back.[/dim]\n")
    return final


def answer_approvals(final: dict) -> dict:
    """Ask at the keyboard for anything the job is holding, and carry on until it's done."""
    while final.get("approval"):
        a = final["approval"]
        if not sys.stdin.isatty():
            console.print("  [yellow]Waiting for approval. Run `cx` interactively, or use the app.[/yellow]\n")
            return final
        prompt = "  [bold]\\[y]es[/bold] / [bold]\\[a]lways in this chat[/bold] / [bold]\\[n]o[/bold]: " if a.get("can_always") \
            else "  [bold]\\[y]es[/bold] / [bold]\\[n]o[/bold]: "
        try:
            reply = console.input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            reply = "n"
        decision = {"y": "approve", "yes": "approve", "a": "approve_session", "always": "approve_session"}.get(reply, "reject")
        if decision == "approve_session" and not a.get("can_always"):
            decision = "approve"
        final = follow(core().resolve_stream(SESSION_ID, a["id"], decision, source="cli"))
    return final


def ask(text: str) -> dict:
    return answer_approvals(follow(core().ask_stream(text, session_id=SESSION_ID, source="cli")))


def _offer_resume():
    """Restart watchers a reboot took down, and offer to carry on any job a crash cut off."""
    import cortex.core.watchers as watchers
    watchers.revive(core().db)
    for job in core().interrupted():
        console.print(f"  [yellow]A job was cut off:[/yellow] {job['text'][:80]}")
        try:
            reply = console.input("  [bold]\\[r]esume[/bold] / [bold]\\[d]ismiss[/bold] / skip: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if reply.startswith("r"):
            answer_approvals(follow(core().resume_stream(job["id"], source="cli")))
        elif reply.startswith("d"):
            core().dismiss_job(job["id"])


def _listen(waiting):
    """Show what a watcher writes into this chat, without waiting to be asked."""
    db = core().db
    last = db.latest_message_id(SESSION_ID)
    while True:
        time.sleep(4)
        if not waiting.is_set():
            last = db.latest_message_id(SESSION_ID)
            continue
        for row in db.recent_messages(SESSION_ID, after_id=last, limit=10):
            last = row["id"]
            if row["role"] == "assistant":
                console.print("\n  [dim]while you were away:[/dim]")
                console.print(Markdown(row["content"]))
                console.print()


def interactive_mode():
    """Main interactive conversation loop."""
    console.print()
    console.print("  [bold]Cortex[/bold] [dim]— type naturally, ctrl+c to quit[/dim]")
    console.print()

    # History
    history_file = os.path.expanduser("~/.cortex_history")
    try:
        if readline:
            readline.read_history_file(history_file)
            readline.set_history_length(500)
    except (FileNotFoundError, PermissionError, OSError):
        pass

    _offer_resume()
    waiting = threading.Event()
    threading.Thread(target=_listen, args=(waiting,), name="cx-updates", daemon=True).start()

    while True:
        try:
            waiting.set()
            user_input = console.input("  [green]→[/green] ").strip()
            waiting.clear()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]bye.[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", "bye"):
            console.print("\n  [dim]bye.[/dim]\n")
            break

        if user_input.lower() in ("voice", "listen", "talk"):
            import cortex.voice.voice_mode as voice_mode
            voice_mode.run_voice_mode()
            continue

        try:
            ask(user_input)
        except Exception as e:
            console.print(f"\n  [red]Error: {e}[/red]\n")

    # Save history
    try:
        if readline:
            readline.write_history_file(history_file)
    except (PermissionError, OSError):
        pass


def one_shot(text: str):
    """Single query, print response, exit."""
    try:
        ask(text)
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]\n")
        sys.exit(1)


def model_cmd(argv: list[str]):
    """`cx model` lists what's available; `cx model <name>` switches, for chat and background."""
    import cortex.llm.llm as llm_mod
    from cortex.util.cortex_bootstrap import load_config
    config = load_config().get("llm", {})
    if argv:
        name = argv[0]
        error = llm_mod.choose_model(name)
        console.print(f"\n  [red]{error}[/red]\n" if error else f"\n  [green]✓[/green] now using {name}\n")
        return
    current = llm_mod.saved_model() or (config.get("ollama") or {}).get("model", "")
    groups = llm_mod.available_models(config)
    for g in groups:
        console.print(f"\n  [dim]{g['label']}[/dim]")
        for m in g["models"]:
            mark = "[green]●[/green]" if m["id"] == current else " "
            lock = "  [dim](needs a key: cx key)[/dim]" if m["needs_key"] else ""
            console.print(f"   {mark} {m['id']}{lock}", highlight=False)
    if not groups:
        console.print("\n  [yellow]No models found. Start Ollama, or add a key with `cx key ollama`.[/yellow]")
    console.print()


def usage_cmd(argv: list[str]):
    """`cx usage` shows cloud use; `cx usage limit <name> <value>` and
    `cx usage price <model> <in> <out>` (USD per 1M tokens) change what's enforced."""
    from cortex.core import usage
    import cortex.storage.settings_store as settings_store
    core()      # binds the database
    if len(argv) >= 3 and argv[0] == "limit":
        settings_store.update_section("limits", {argv[1]: float(argv[2])})
        console.print(f"\n  [green]✓[/green] {argv[1]} = {argv[2]} (0 means no limit)\n")
        return
    if len(argv) >= 4 and argv[0] == "price":
        prices = dict(settings_store.section("limits").get("prices") or {})
        prices[argv[1]] = [float(argv[2]), float(argv[3])]
        settings_store.update_section("limits", {"prices": prices})
        console.print(f"\n  [green]✓[/green] {argv[1]}: ${argv[2]} in, ${argv[3]} out per million tokens\n")
        return
    r = usage.report()
    console.print(f"\n  [bold]Cloud use:[/bold] ${r['day']['usd']:.2f} today ({r['day']['tokens']:,} tokens), "
                  f"${r['month']['usd']:.2f} this month ({r['month']['tokens']:,} tokens)")
    console.print("  [dim]Limits (0 = none): " + ", ".join(f"{k}={v}" for k, v in r["limits"].items()) + "[/dim]")
    for m in r["models"]:
        cost = "free" if m["provider"] == "local" else ("not priced" if m["cost_usd"] is None else f"${m['cost_usd']:.2f}")
        console.print(f"   {m['model']:<34} {m['calls']:>5} calls  "
                      f"{(m['input_tokens'] or 0) + (m['output_tokens'] or 0):>11,} tokens  {cost}", highlight=False)
    console.print()


def undo_cmd(argv: list[str]):
    """`cx undo` puts back what the last job changed, or `cx undo <job id>`."""
    db = core().db
    job = argv[0] if argv else db.last_undoable_any()
    console.print(f"\n  {core().undo.undo_job(job)['message'] if job else 'There is nothing to undo.'}\n")


def watches_cmd(argv: list[str]):
    """`cx watches` lists what's being looked after; `cx watches stop <id>` ends one."""
    import cortex.core.watchers as watchers
    db = core().db
    if len(argv) >= 2 and argv[0] == "stop":
        console.print("\n  [green]✓[/green] stopped\n" if watchers.stop(db, argv[1]) else "\n  no such watch\n")
        return
    rows = watchers.active(db)
    console.print()
    for w in rows:
        console.print(f"   {w['id']}  {w['label']}", highlight=False)
    console.print("  [dim]nothing is being watched[/dim]\n" if not rows else "")


def key_cmd(argv: list[str]):
    """`cx key <provider>` stores an API key in the OS keychain. It's typed at a hidden
    prompt rather than passed as an argument, so it never lands in shell history."""
    import getpass
    import cortex.llm.llm as llm_mod
    provider = (argv[0].lower() if argv else "")
    if provider not in llm_mod.PROVIDER_LABEL:
        console.print(f"\n  usage: cx key <{'|'.join(llm_mod.PROVIDER_LABEL)}>"
                      "\n  An Ollama Cloud key comes from https://ollama.com/settings/keys\n")
        return
    key = getpass.getpass(f"  {llm_mod.PROVIDER_LABEL[provider]} key: ").strip()
    if key:
        ok = llm_mod.set_key(provider, key)
        console.print("\n  [green]✓[/green] saved. Pick a model with `cx model`.\n" if ok
                      else "\n  [red]Couldn't save the key.[/red]\n")


def memory_cmd(argv: list[str]):
    """
    `cx memory [list [type] | add [rule|fact|preference|project_info] <text> | forget <id>]`

    Direct read/write access to Cortex's long-term memory — no conversation,
    no daemon, no dependence on it noticing or remembering to extract a fact.
    `rule` entries are standing instructions Cortex always obeys (see
    cortex/core/memory.py / safety.py) — use this to add or remove one
    yourself instead of just hoping a correction "took" in chat.
    """
    from cortex.util.cortex_bootstrap import load_config, connect_db
    db = connect_db(load_config())
    if db is None:
        console.print("\n  [red]Could not open Cortex's database.[/red]\n")
        return

    sub = (argv[0].lower() if argv else "list")

    if sub == "list":
        type_filter = argv[1] if len(argv) > 1 else None
        mems = db.get_memories(type=type_filter, limit=200)
        if not mems:
            console.print("\n  [dim](no memories stored)[/dim]\n")
            return
        console.print()
        for m in mems:
            tag = "[bold yellow]RULE[/bold yellow]" if m["type"] == "rule" else f"[dim]{m['type']}[/dim]"
            console.print(f"  #{m['id']:<4} {tag}  {m['content']}")
        console.print()

    elif sub == "add":
        rest = argv[1:]
        mtype = "fact"
        if rest and rest[0].lower() in ("rule", "fact", "preference", "project_info"):
            mtype, rest = rest[0].lower(), rest[1:]
        content = " ".join(rest).strip()
        if not content:
            console.print("\n  usage: cx memory add [rule|fact|preference|project_info] <text>\n")
            return
        from cortex.core.memory import MemoryManager
        result = MemoryManager(db, None, {}).store_fact(content, mem_type=mtype, source="cx-cli",
                                                        importance=5 if mtype == "rule" else 4)
        if result["status"] == "secret":
            console.print("\n  [yellow]That looks like a password or key, so I won't keep it.[/yellow]\n")
        else:
            console.print(f"\n  [green]✓[/green] remembered ({mtype}): {content}\n")

    elif sub == "forget":
        if len(argv) < 2 or not argv[1].isdigit():
            console.print("\n  usage: cx memory forget <id>   (see `cx memory list` for ids)\n")
            return
        mid = int(argv[1])
        ok = db.delete_memory(mid)
        if ok:
            console.print(f"\n  [green]✓[/green] forgot #{mid}\n")
        else:
            console.print(f"\n  [yellow]no memory #{mid} found[/yellow]\n")

    else:
        console.print("\n  usage: cx memory [list [type] | add [type] <text> | forget <id>]\n")


def workers_cmd(argv: list[str]):
    """
    `cx workers [list | on <name> | off <name> | run <name> | profile <off|essential|balanced|full>
                 | pause | resume | free]`

    Control Cortex's own background workers — the scheduled jobs that run when you're not
    chatting — and the AI model's memory. `<name>` is a worker key or its name
    (autonomy, reminders, "morning briefing", collector.system, task.<id> …).
    """
    import cortex.core.workers as workers
    import cortex.core.resources as resources
    sub = (argv[0].lower() if argv else "list")
    arg = " ".join(argv[1:]).strip()

    def show():
        rows = workers.list_workers()
        s = workers.summary(rows)
        console.print(f"\n  [bold]Background:[/bold] profile [cyan]{s['profile']}[/cyan] · "
                      f"{s['on']}/{s['total']} workers on · {s['ai_on']} AI job(s) on · "
                      f"{s['tasks_on']}/{s['tasks_total']} tasks on")
        for gid, gname, _ in workers.GROUPS:
            grp = [r for r in rows if r["group"] == gid]
            if not grp:
                continue
            console.print(f"\n  [dim]{gname}[/dim]")
            for r in grp:
                dot = "[green]●[/green]" if r["on"] else "[dim]○[/dim]"
                cost = {"light": "[dim]light[/dim]", "medium": "[yellow]medium[/yellow]", "heavy": "[red]heavy[/red]"}[r["cost"]]
                console.print(f"   {dot} {r['key']:<20} {r['name']:<22} {r['cadence']:<14} {cost}")
        console.print(f"\n  [dim]{resources.describe()}[/dim]\n")

    if sub in ("list", "ls", "status"):
        show()
    elif sub in ("on", "off", "enable", "disable", "run"):
        if not arg:
            console.print(f"\n  usage: cx workers {sub} <name>   (see `cx workers`)\n")
            return
        key = workers.resolve_key(arg)
        if not key:
            console.print(f"\n  [yellow]no worker matches '{arg}'[/yellow] — run `cx workers` to see them\n")
            return
        if sub == "run":
            r = workers.run_now(key)
        else:
            r = workers.set_enabled(key, sub in ("on", "enable"))
        mark = "[green]✓[/green]" if r.get("ok") else "[red]✗[/red]"
        console.print(f"\n  {mark} {key}" + (f" — {r['error']}" if r.get("error") else "") + "\n")
    elif sub == "profile":
        if arg not in workers.PROFILES:
            console.print(f"\n  usage: cx workers profile <{'|'.join(workers.PROFILES)}>\n")
            return
        r = workers.apply_profile(arg)
        console.print(f"\n  profile → [cyan]{r.get('profile')}[/cyan]"
                      + ("".join(f"\n  [yellow]{e}[/yellow]" for e in r.get("errors", []))) + "\n")
    elif sub == "pause":
        r = workers.apply_profile("off")
        console.print(f"\n  [green]✓[/green] all background work paused (profile: {r.get('profile')})\n")
    elif sub == "resume":
        r = workers.apply_profile("essential")
        console.print(f"\n  [green]✓[/green] resumed essentials — reminders + rules "
                      f"(use `cx workers profile balanced|full` for more)\n")
    elif sub in ("free", "unload"):
        console.print(f"\n  {resources.unload_models()}\n")
    else:
        console.print("\n  usage: cx workers [list | on <name> | off <name> | run <name> | "
                      "profile <name> | pause | resume | free]\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("setup", "hello", "onboard"):
        from cortex.core.onboarding import run_onboarding
        run_onboarding(console)
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() in ("gui", "ui", "web"):
        console.print("\n  [dim]Opening Cortex GUI at http://127.0.0.1:7860 … (Ctrl+C to stop)[/dim]\n")
        import cortex.server.gui_server as gui_server
        gui_server.main()
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() in ("voice", "listen", "talk"):
        import cortex.voice.voice_mode as voice_mode
        voice_mode.run_voice_mode()
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() == "memory":
        memory_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() in ("model", "models"):
        model_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() == "key":
        key_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() == "usage":
        usage_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() == "undo":
        undo_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() in ("watches", "watch"):
        watches_cmd(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1].lower() in ("workers", "worker", "bg", "background"):
        workers_cmd(sys.argv[2:])
        return

    # Warm first-run — only in interactive mode (not for one-shot `cx "..."`).
    if len(sys.argv) <= 1 and sys.stdin.isatty():
        try:
            from cortex.core.onboarding import needs_onboarding, run_onboarding
            if needs_onboarding():
                run_onboarding(console)
        except Exception:
            pass

    if len(sys.argv) > 1:
        one_shot(" ".join(sys.argv[1:]))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
