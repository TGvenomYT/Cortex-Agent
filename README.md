# Cortex ✦ — your brother in the machine

A private, on-device AI assistant. It runs a **local LLM** (via [Ollama](https://ollama.com)),
keeps all its data in a **local SQLite database**, and gives you a clean chat GUI that can
actually *do* things on your machine — answer questions about your system, run commands,
read your email, search the web, make charts, remember what matters, and quietly run
recurring tasks for you.

No cloud keys. No account. Nothing leaves your machine unless you ask it to.

Runs on **macOS, Windows and Linux**.

---

## Quick start

**One command, from the project folder** — it sets up everything:

| OS | Command | or just… |
|----|---------|----------|
| macOS | `python3 run.py` | double-click `start.command` |
| Linux | `python3 run.py` | `./start.sh` |
| Windows | `py run.py` | double-click `start.bat` |

`run.py` is a **dynamic installer**. It detects your OS and, in order:

1. creates a local virtualenv and installs the Python requirements,
2. installs **Ollama** (the local LLM engine) if it's missing — via Homebrew / the
   official Linux script / winget — and starts it,
3. checks the model in `config.yaml`; if it isn't available on your machine it pulls a
   sensible default (`gemma3:12b`) and points the config at it,
4. opens the Cortex GUI at **http://127.0.0.1:7860**, whose first run walks you through
   onboarding (your name, optional server).

Everything is incremental — anything already installed is left alone, so re-running is
safe and a machine that's already set up just launches.

```
python3 run.py --voice        # also install the optional voice/audio extras
python3 run.py --yes          # non-interactive: accept the recommended defaults
python3 run.py --skip-ollama  # just venv + launch, don't touch Ollama/models
```

> You can still do it by hand: install Ollama from [ollama.com](https://ollama.com),
> `ollama pull <model>`, set it in `config.yaml`, then `python3 run.py --skip-ollama`.

---

## What it can do

- **Chat that acts** — a multi-step agent loop: it plans, runs tools, looks at the
  results, and answers in plain language (never dumps raw logs at you).
- **Knows your system** — ask “is my machine OK?” and it actually checks.
- **Recurring tasks** — “every morning at 8, email me a news briefing.” Cortex schedules
  it with your OS scheduler (launchd / cron / Task Scheduler) and delivers the result.
- **Proactive nudges** — a notification bell for things it noticed while you were away.
- **Email** — connect Gmail (app password, stored in your OS keychain) to triage your inbox.
- **Web + vision + charts** — web search, page reading, image search, and a `make_chart`
  tool that renders real plots inline.
- **Memory** — remembers your preferences and facts across sessions.
- **Attachments / RAG** — drop in PDFs, docs, images or video and ask about them.
- **Works offline** — the whole thing runs against your local LLM and local DB.

---

## Project layout

```
Cortex Agent/
  run.py            one-command installer + launcher
  run_once.py       thin shim for scheduled jobs (launchd/cron/schtasks call this)
  uninstall.py      cross-platform uninstaller
  config.yaml       models, timeouts, feature toggles
  cortex/           the application package
    core/           agent loop, memory, safety, reminders, automation, autonomy, tasks
    llm/            Ollama client, tools, schemas, embeddings, chart + NLG
    web/            web search, crawl, browser automation
    voice/          MLX voice server + voice mode
    storage/        SQLite + Postgres-sync backends, RAG, settings
    server/         Flask GUI + the scheduled-job runner
    integrations/   Gmail, calendar, MCP client
    cli/            the `cx` terminal app + dashboard
    util/           OS compat, secrets, config bootstrap, notifications
  collectors/       background system-state collectors (run on a schedule)
  assets/           app icons + favicon
  docs/             plan, DB schema, Siri shortcuts
  legacy/           archived old daemon (not used — see legacy/README.md)
```

## Where your data lives

Everything is under `~/.cortex/`:

| Path | What |
|------|------|
| `~/.cortex/cortex.db` | the local SQLite database (conversations, memory, tasks, …) |
| `~/.cortex/settings.json` | your settings (name, connected accounts, scheduled tasks) |
| `~/.cortex/artifacts/` | generated charts/images |
| OS secret store | DB/email passwords (macOS Keychain · Windows Credential Manager · Linux Secret Service) |

Nothing here is sent anywhere. Secrets are never written to plaintext config.

---

## Platform notes

- **macOS** is the most battle-tested (notifications via Notification Center, always-on
  recurring tasks via `launchd`, MLX-accelerated voice).
- **Linux**: notifications via `notify-send`, recurring tasks via `cron`.
- **Windows**: toast notifications via PowerShell, recurring tasks via Task Scheduler.
- All OS-specific behaviour lives in one file — `osutil.py` — and degrades gracefully
  (if a machine can't notify, Cortex logs instead of crashing).

## Cloud models (optional, plug-and-play)

Cortex runs 100% locally by default. To also use a frontier cloud model, open
**Settings → AI Providers** and paste an API key for **Claude (Anthropic)**, **OpenAI**,
or **Gemini (Google)**. Then pick its model from the selector at the top-left. Keys are
stored in your OS keychain (macOS Keychain / Windows Credential Manager / Linux Secret
Service) — never in a file, and only ever sent to that provider. Remove a key any time
from the same screen and Cortex falls back to local.

Everything works the same across providers — the agent loop, tools, charts, PDF search —
because they all go through one unified interface in `cortex/llm/llm.py`.

## Configuration

Edit `config.yaml` to pick your default local model(s), timeouts, and feature toggles.
The default backend is local SQLite — no setup needed. Optional Postgres sync (to a
self-hosted Supabase) is available but off by default.

`cx gui` serves the app and prints the URL but does **not** auto-open a browser; the
one-command installer (`run.py`) opens it once on first run.

## Uninstall

```bash
python3 uninstall.py            # macOS / Linux   (or double-click uninstall.command)
py uninstall.py                 # Windows         (or double-click uninstall.bat)
```

It removes Cortex's scheduled jobs, stored secrets, the `cx` shim and the virtualenv,
but **keeps your data and Ollama by default**. Options:

```
python3 uninstall.py --dry-run         # show what it would remove, change nothing
python3 uninstall.py --purge           # also delete your data in ~/.cortex (irreversible)
python3 uninstall.py --remove-ollama   # also uninstall Ollama + its models
python3 uninstall.py --yes             # non-interactive, safe defaults
```

The project folder itself is left in place — delete it by hand when you're done.

## License

MIT
