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
try:
    import readline  # nicer line editing on macOS/Linux; absent on Windows
except ImportError:
    readline = None
import urllib.request
import urllib.error
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

BASE_URL = "http://localhost:7800"
SESSION_ID = f"cli-{datetime.now().strftime('%Y%m%d-%H%M')}"

# Force a reasonable width even when piped
console = Console(width=100 if not sys.stdout.isatty() else None)


def post(endpoint: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    # Local Ollama inference (gemma4:e4b) can take 10-30s+ depending on prompt
    # size, vs 2-3s for a cloud API. 120s gives headroom without hanging forever.
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def is_daemon_running() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/status")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _is_viz_request(text: str) -> bool:
    """Detect if the user is asking for a chart/graph/visualization."""
    viz_keywords = ["chart", "graph", "visualize", "visualization", "bar chart", "sparkline",
                    "plot", "show me a graph", "usage chart", "metrics chart"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in viz_keywords)


def _is_todo_request(text: str) -> bool:
    """Detect if the user is asking about todos/tasks."""
    todo_keywords = ["todo", "task", "add to my list", "my tasks", "to-do", "to do list",
                     "what's on my list", "mark as done", "complete task"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in todo_keywords)


def _handle_todo(text: str):
    """Handle todo requests via AppFlowy API."""
    text_lower = text.lower()

    try:
        if any(kw in text_lower for kw in ["add", "create", "new task", "put"]):
            # Extract the task title — everything after the action keyword
            import re
            title = re.sub(r'^(add|create|put|new task)[:\s]*(to my (todo|task|list)[s]?[:\s]*)?', '', text, flags=re.IGNORECASE).strip()
            if not title:
                title = text
            result = post("/todos", {"title": title})
            console.print(f"\n  [green]✓[/green] {result.get('speak', 'Added.')}\n")

        elif any(kw in text_lower for kw in ["list", "show", "what", "pending", "my task", "my todo"]):
            result = _get_json("/todos")
            formatted = result.get("formatted", "No tasks.")
            count = result.get("count", 0)
            console.print(f"\n  [bold]Tasks ({count}):[/bold]")
            for line in formatted.splitlines():
                console.print(f"  {line}")
            console.print()

        else:
            # Let the LLM handle ambiguous todo requests
            response = send_message(text)
            display_response(response)

    except urllib.error.HTTPError as e:
        if e.code == 400:
            body = json.loads(e.read())
            console.print(f"\n  [yellow]{body.get('speak', 'AppFlowy not configured.')}[/yellow]\n")
        else:
            console.print(f"\n  [red]AppFlowy error: {e}[/red]\n")
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]\n")


def _get_json(endpoint: str) -> dict:
    """GET request to Cortex API."""
    url = f"{BASE_URL}{endpoint}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _handle_screen(question: str):
    """Capture screen and get Cortex to explain it."""
    try:
        if sys.stdin.isatty():
            with console.status("[dim]capturing screen...[/dim]", spinner="dots"):
                result = post("/vision", {"question": question, "source": "cli"})
        else:
            result = post("/vision", {"question": question, "source": "cli"})

        speak = result.get("speak", "")
        if speak:
            console.print()
            from rich.markdown import Markdown
            console.print(Markdown(speak))
            console.print()
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]\n")


_CORE = None  # in-process Cortex, created on first offline use


def _inprocess_ask(text: str) -> dict:
    """Run the ask-pipeline in-process (no server) — works fully offline."""
    global _CORE
    if _CORE is None:
        from cortex.core.cortex_core import CortexCore
        _CORE = CortexCore()
    return _CORE.ask(text, session_id=SESSION_ID, source="cli")


def send_message(text: str) -> dict:
    # Local-first: if the on-demand server isn't up, just think in-process.
    # No daemon required to talk to Cortex anymore.
    if is_daemon_running():
        return post("/ask", {"text": text, "source": "cli", "session_id": SESSION_ID})
    return _inprocess_ask(text)


def send_message_streaming(text: str):
    """
    Send message to streaming endpoint and yield response chunks in real-time.
    Returns the full accumulated response at the end.
    """
    body = json.dumps({
        "text": text,
        "source": "cli",
        "session_id": SESSION_ID
    }).encode()
    
    req = urllib.request.Request(
        f"{BASE_URL}/ask/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    full_text = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                if "chunk" in data:
                    full_text += data["chunk"]
                    yield data["chunk"]
                elif data.get("done"):
                    # Stream complete — return the full text for JSON parsing
                    full_text = data.get("full_text", full_text)
                    break
                elif "error" in data:
                    yield f"\n[error: {data['error']}]"
                    break
    except Exception as e:
        yield f"\n[error: {e}]"
    
    return full_text


def display_response(response: dict):
    """Render the response as markdown in terminal."""
    speak = response.get("speak", "")
    exec_result = response.get("execution_result")

    if not speak:
        console.print("[dim]  (no response)[/dim]")
        return

    # Render the explanation as markdown
    console.print()
    md = Markdown(speak)
    console.print(md)

    # Only show raw error output if command failed
    if exec_result:
        exit_code = exec_result.get("exit_code", 0)
        stderr = exec_result.get("stderr", "").strip()

        if exit_code != 0 and stderr:
            console.print()
            console.print(Panel(
                stderr[:500],
                title="[red]error[/red]",
                border_style="red"
            ))

    # Show reminder confirmation
    if response.get("reminder_id"):
        due = response.get("due", "?")
        console.print(f"  [dim]⏰ reminder #{response['reminder_id']} → {due}[/dim]")

    console.print()


def interactive_mode():
    """Main interactive conversation loop."""
    console.print()
    console.print("  [bold]Cortex[/bold] [dim]— type naturally, ctrl+c to quit[/dim]")
    console.print()

    # History
    history_file = os.path.expanduser("~/.cortex_history")
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    readline.set_history_length(500)

    while True:
        try:
            user_input = console.input("  [green]→[/green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]bye.[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", "bye"):
            console.print("\n  [dim]bye.[/dim]\n")
            break

        # Special commands
        if user_input.lower() in ("dashboard", "dash", "stats", "viz"):
            from cortex.cli.dashboard import render_dashboard
            render_dashboard()
            continue

        if user_input.lower() in ("voice", "listen", "talk"):
            from cortex.voice.voice import voice_loop
            voice_loop()
            continue

        if user_input.lower() in ("screen", "look", "what is this", "whats on screen"):
            _handle_screen(user_input)
            continue

        # Intercept visualization requests — handle client-side
        if _is_viz_request(user_input):
            from cortex.cli.dashboard import render_chart
            render_chart(user_input)
            continue

        # Intercept todo keywords — route to AppFlowy
        if _is_todo_request(user_input):
            _handle_todo(user_input)
            continue

        try:
            if is_daemon_running():
                console.print()
                full_text = ""
                for chunk in send_message_streaming(user_input):
                    print(chunk, end="", flush=True)
                    full_text += chunk
                print()
                try:
                    speak = json.loads(full_text).get("speak", full_text)
                    if speak:
                        console.print(); console.print(Markdown(speak))
                except json.JSONDecodeError:
                    pass
                console.print()
            else:
                with console.status("[dim]thinking…[/dim]", spinner="dots"):
                    response = _inprocess_ask(user_input)
                display_response(response)
        except Exception as e:
            console.print(f"\n  [red]Error: {e}[/red]\n")

    # Save history
    try:
        readline.write_history_file(history_file)
    except (PermissionError, OSError):
        pass


def one_shot(text: str):
    """Single query, print response, exit."""
    try:
        if is_daemon_running():
            console.print()
            full_text = ""
            for chunk in send_message_streaming(text):
                print(chunk, end="", flush=True)
                full_text += chunk
            print()
            try:
                speak = json.loads(full_text).get("speak", full_text)
                if speak and speak != full_text:
                    console.print(); console.print(Markdown(speak))
            except json.JSONDecodeError:
                pass
            console.print()
        else:
            with console.status("[dim]thinking…[/dim]", spinner="dots"):
                response = _inprocess_ask(text)
            display_response(response)
    except Exception as e:
        console.print(f"\n  [red]Error: {e}[/red]\n")
        sys.exit(1)


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
        db.save_memory(type=mtype, content=content, source="cx-cli",
                       importance=5 if mtype == "rule" else 4)
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
    # Local-first: no daemon required. If the on-demand server is up we use it
    # (streaming); otherwise Cortex thinks in-process, fully offline.
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
