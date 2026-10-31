# Cortex ✦ — your brother in the machine

A private AI assistant that lives on your computer. It runs a **local LLM** (via
[Ollama](https://ollama.com)) or a **cloud model** if you add a key, keeps all its data in a
**local SQLite database**, and gives you a clean chat GUI that can actually *do* things on
your machine — answer questions about your system, run commands, read your email, search
the web, make charts, remember what matters, and quietly run recurring tasks for you.

No account needed. With a local model, nothing leaves your machine unless you ask it to.

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
4. adds a `cx` command (on macOS and Linux, in `~/.local/bin`),
5. opens the Cortex GUI at **http://127.0.0.1:7860**, whose first run walks you through
   setup: your name, an optional server, and what Cortex may keep track of.

Everything is incremental — anything already installed is left alone, so re-running is
safe and a machine that's already set up just launches.

```
python3 run.py --voice        # also install the optional voice/audio extras
python3 run.py --yes          # non-interactive: accept the recommended defaults
python3 run.py --skip-ollama  # just venv + launch, don't touch Ollama/models
```

> You can still do it by hand: install Ollama from [ollama.com](https://ollama.com),
> `ollama pull <model>`, set it in `config.yaml`, then `python3 run.py --skip-ollama`.

> **Can't install Ollama?** You don't need it. Add an **Ollama Cloud** key (see
> [Cloud models](#cloud-models-optional-plug-and-play)) and pick a cloud model. Uploaded files
> are then searched by keyword rather than by meaning.

---

## What it can do

- **Chat that acts** — a multi-step agent loop: it plans, runs tools, looks at the
  results, and answers in plain language (never dumps raw logs at you). Replies appear as
  they're written.
- **Asks before it acts** — see [Approvals](#approvals) below.
- **Knows your system** — ask “is my machine OK?” and it actually checks.
- **Recurring tasks** — “every morning at 8, email me a news briefing.” Cortex schedules
  it with your OS scheduler (launchd / cron / Task Scheduler) and delivers the result. A run
  with nothing to say stays quiet, and a job with an end (“tell me when X happens”) stops
  itself once it's done.
- **Proactive nudges** — a notification bell for things it noticed while you were away.
- **Email** — connect Gmail (app password, stored in your OS keychain) to triage your inbox.
- **Web + vision + charts** — web search, page reading, image search, and a `make_chart`
  tool that renders real plots inline.
- **Memory** — picks up what you tell it about yourself as you talk, and remembers it in
  every later chat. See and delete it all in Settings → Memory.
- **Attachments / RAG** — drop in PDFs, docs, images or video and ask about them.
- **Works offline** — with a local model, the whole thing runs against your local LLM and
  local DB.

## Approvals

Every action Cortex takes goes through one gate (`cortex/core/safety.py`), whether it comes
from chat, a helper, a scheduled task or autonomy:

- **Read-only work runs on its own.** A command counts only if every part of it is known to
  be read-only. Anything unrecognised is treated as not read-only.
- **Everything else asks first**, with buttons, and the job carries on from where it paused
  once you answer. "Allow in this chat" stops the prompts for ordinary repeated commands.
- **When nobody's there** (scheduled tasks, autonomy), it goes to the approvals inbox in the
  notification bell, and the job continues without it.
- **Some things are refused outright**, even if you approve: erasing disks, deleting a home
  or system folder, force-pushing, and reading saved passwords.

Once Cortex has read outside content in a turn (a web page, an email, a file), that text might
contain instructions from a stranger. It's marked as untrusted, and from then on everything
beyond reading asks first, whatever you allowed earlier.

Settings → Background → Approvals can loosen this to **Balanced**, where ordinary commands you
asked for run without a prompt. Destructive ones always ask.

---

## Running jobs

Every job is recorded as it goes, so it isn't lost with the app.

- **Stop** a job with the stop button (Ctrl+C in `cx`). That also stops every process it started.
- **Undo** what a job changed with the button under its reply, by asking, or with `cx undo`. Cortex
  keeps a copy before it changes files, for 14 days. Approval prompts say up front when something
  can't be undone: a script, an installer, or anything on another machine.
- **Crashes:** if Cortex dies mid-job, the chat offers to carry on from the last finished step
  (`cx` offers on start).
- **Stuck jobs:** a model that goes silent is cut off and tried once more, a job repeating itself or
  failing over and over is ended early, and it tells you what happened rather than claiming success.
- **Long commands** don't get killed. After a minute a command moves to the background and keeps going.

### Looking after something that's running

Ask "tell me when the build finishes" or "keep an eye on that server". A small separate process
watches it, using no AI while it waits, and keeps going if you close the app. When something
happens, Cortex writes the result into that chat by itself and raises a notification, with the
reason if it failed. It can watch a process, a port, a log file, or any read-only command, and
it warns once if a log goes quiet for 15 minutes. See them with `cx watches` or in Settings → Background.

### Usage limits

Settings → Usage shows what cloud models have used and sets limits for the day, the month and a
single job (`cx usage`, `cx usage limit daily_usd 10`). Dollar figures are estimates from the
providers' published prices. Ollama Cloud has no price table here, so it's held by the token
limits unless you add one (`cx usage price <model> <in> <out>`). Past a limit, cloud calls stop
and a local model takes over if there is one. Local models are free and never limited.

### Context

Cortex reads each model's real limits from the model itself where it can (Ollama, Anthropic,
Gemini), and sizes history and tool output to fit. As a job fills up it shortens old results, then
summarises earlier work, and only then drops the oldest steps. Long chats are folded into a summary
between turns. The meter under the input shows how full the context is. Cloud models are capped at
`agent.max_context_tokens` in `config.yaml` (default 100,000), because every token is paid for on every step.

## Project layout

```
Cortex Agent/
  run.py            one-command installer + launcher
  run_once.py       thin shim for scheduled jobs (launchd/cron/schtasks call this)
  uninstall.py      cross-platform uninstaller
  config.yaml       models, timeouts, feature toggles
  cortex/           the application package
    core/           agent loop, the safety gate, job runtime, undo, watchers, usage limits,
                    context management, memory, reminders, automation, autonomy, tasks
    llm/            Ollama client, tools, schemas, embeddings, chart + NLG
    web/            web search, crawl, browser automation
    voice/          MLX voice server + voice mode
    storage/        SQLite + Postgres-sync backends, RAG, settings
    server/         Flask GUI + the scheduled-job runner
    integrations/   Gmail, calendar, MCP client
    cli/            the `cx` terminal app
    util/           OS compat, secrets, config bootstrap, notifications
  collectors/       background system-state collectors (run on a schedule)
  assets/           app icons + favicon
  docs/             plan, DB schema, Siri shortcuts
  tests/            pytest suite (pip install -r requirements-dev.txt, then pytest)
  legacy/           archived old daemon (not used — see legacy/README.md)
```

## Your data, on your machine

Each person's install is entirely their own: their name, memory, chats and settings live in
their `~/.cortex`, and nothing is shared between machines.

**Memory.** After each message Cortex notes what you said about yourself, reading only your own
words. A correction replaces the old note, repeats are ignored, and "from now on…" becomes a
standing rule. Each question brings up whatever memories bear on it, so nothing is lost as they
add up. **Passwords, keys and card numbers are never kept**, and are masked in stored chats.

**What it keeps track of.** Nothing runs in the background until setup asks. Reminders, this
computer's health and your projects are on by default; shell history and machine details (installed
apps, Wi-Fi names, SSH key names) stay off unless you turn them on. Change it any time in Settings →
Background or with `cx workers`. If you only use cloud models, these stay on your machine, but what's
relevant to a question goes to that provider with it.

**Your own settings** go in `~/.cortex/config.yaml`, which sits on top of the project's `config.yaml` and
survives updates. Keep the project's copy free of anything personal, since everyone shares it.

## Where your data lives

Everything is under `~/.cortex/`:

| Path | What |
|------|------|
| `~/.cortex/cortex.db` | the local SQLite database (conversations, memory, tasks, …) |
| `~/.cortex/settings.json` | your settings (name, connected accounts, scheduled tasks) |
| `~/.cortex/config.yaml` | your own overrides of the shared config (optional) |
| `~/.cortex/artifacts/` | generated charts/images |
| `~/.cortex/checkpoints/` | copies kept for undo (14 days) |
| `~/.cortex/runs/` | output of background commands (3 days) |
| OS secret store | DB/email passwords (macOS Keychain · Windows Credential Manager · Linux Secret Service) |

Nothing here is sent anywhere, except that a cloud model sees what it needs for the task
at hand (your messages, relevant memories, uploaded text) when you choose to use one. Secrets
are never written to plaintext config.

---

## Platform notes

- **macOS** is the most battle-tested (notifications via Notification Center, background
  work via `launchd`, MLX-accelerated voice).
- **Linux**: notifications via `notify-send`, background work via `cron`.
- **Windows**: toast notifications via PowerShell, background work via Task Scheduler.
- All OS-specific behaviour lives in one file — `osutil.py` — and degrades gracefully
  (if a machine can't notify, Cortex logs instead of crashing).

## Cloud models (optional, plug-and-play)

Cortex runs locally by default. To also use a cloud model, open **Settings → AI Providers**
(or run `cx key <provider>`) and paste a key, then pick the model from the selector at the
top-left (or `cx model <name>`). Your choice is remembered, and the terminal and background
jobs use it too.

| Provider | Key from | Needs Ollama installed? |
|----------|----------|-------------------------|
| **Ollama Cloud** | [ollama.com/settings/keys](https://ollama.com/settings/keys) | No |
| Claude | [console.anthropic.com](https://console.anthropic.com/settings/keys) | No |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | No |
| Gemini | [aistudio.google.com](https://aistudio.google.com/apikey) | No |

**How Ollama is routed:** local models go to the Ollama running on this machine
(`localhost:11434`), and cloud models go straight to `https://ollama.com/api` with your key.
The cloud list needs no key, so it shows up even with nothing installed. Models already
signed in through `ollama signin` keep working through the local daemon if you add no key.

If you've added a key and the local engine isn't running, a job fails over to a cloud model
instead of stopping. Keys are stored in your OS keychain — never in a file, and only ever sent
to that provider. Remove one any time from the same screen.

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
