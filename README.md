# Cortex — The Control Layer

A central AI daemon that acts as the intelligence layer for your macOS workstation. Always-on, always aware.

```
$ cx

  Cortex — type naturally, ctrl+c to quit

  → any crashed containers?
  Your hotel_mapping_worker_v2 got OOM-killed 8 days ago — exit code 137.
  The celery worker ran out of memory. Consider bumping the limit.

  → show cpu chart
  ╭──────────── CPU Usage ────────────╮
  │  CPU:  ██░░░░░░░░░░░░░░░░░  8.2%  │
  ╰────────────────────────────────────╯

  → remind me to deploy at 5pm
  ⏰ reminder #3 → 2026-08-07T17:00:00
```

## What It Does

- **Pre-computed awareness** — collectors gather system state every 1-5 min, stored in Postgres. Cortex already knows your system state before you ask.
- **Natural conversation** — talk to your machine. Ask questions, run commands, set reminders, deploy services.
- **Siri integration** — speak to Cortex via Siri Shortcuts. Responses are natural spoken language.
- **DevOps agent** — multi-step task execution with error recovery and rollback.
- **Terminal visualizations** — bar charts, container status tables, system metrics via Rich.
- **Persistent memory** — learns your preferences, tools, workflows across sessions.
- **Multi-scope queries** — complex questions fan out parallel subprocess calls for fast comprehensive answers.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CORTEX DAEMON (:7800)               │
│  Flask + LLM Router + Executor + Memory + Reminders │
└────────────────────────┬────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
    │ Context │    │  OpenAI   │   │  Postgres  │
    │Collectors│   │ GPT-4o-m  │   │ (Supabase) │
    │ (6 jobs) │   │ + Ollama  │   │            │
    └─────────┘    └───────────┘   └────────────┘
```

## Quick Start

```bash
# 1. Clone
git clone <repo> && cd cortex

# 2. Install
./install.sh

# 3. Set env vars (add to ~/.zshrc)
export CORTEX_DB_PASSWORD="your-password"
export OPENAI_API_KEY="sk-..."

# 4. Apply schema
CORTEX_DB_PASSWORD=... python3 test_connection.py

# 5. Start
cx
```

## Components

| File | Purpose |
|------|---------|
| `cortex.py` | Main Flask daemon — all API endpoints |
| `cx` | Interactive CLI (Rich-powered) |
| `db.py` | PostgreSQL connection pool + all DB operations |
| `llm.py` | OpenAI + Ollama backends with fallback router |
| `executor.py` | Shell command execution + logging |
| `context_reader.py` | Builds LLM prompts with pre-computed context |
| `memory.py` | Persistent fact extraction + retrieval |
| `reminders.py` | Background reminder thread + macOS notifications/TTS |
| `devops_agent.py` | Multi-step task planning with error recovery |
| `multi_scope.py` | Parallel fan-out for complex queries |
| `dashboard.py` | Terminal visualizations (Rich bar charts, tables) |
| `collectors/` | 6 launchd-scheduled context collectors |

## Collectors

| Collector | Interval | Data |
|-----------|----------|------|
| shell | 60s | zsh history, recent commands |
| process | 60s | docker containers, ports, services |
| git | 120s | branch, uncommitted files, last commit |
| system | 300s | disk, RAM, CPU, uptime, network |
| project | 30min | project type, deps, structure, TODOs |
| metrics | 60s | Prometheus data (CPU, memory, containers) |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ask` | POST | Main intelligence (text → action) |
| `/execute` | POST | Direct command execution |
| `/remind` | POST | Natural language reminders |
| `/reminders` | GET | List active reminders |
| `/devops` | POST | Multi-step task planning |
| `/devops/execute` | POST | Execute approved plan |
| `/status` | GET | Daemon health check |
| `/memory` | GET/POST | Query/add memories |
| `/projects/track` | POST | Register a project |
| `/conversations` | GET | Session history |
| `/reload` | POST | Hot-reload daemon |

## Siri Integration

See `siri/setup-guide.md` for Shortcut configuration. The main shortcut calls `/ask` with `source: "siri"` which activates voice mode — natural conversational responses optimized for being spoken aloud.

## Requirements

- macOS
- Python 3.12+
- PostgreSQL (Supabase works)
- OpenAI API key (GPT-4o-mini)
- Optional: Ollama for local fallback, Prometheus for metrics

## License

MIT
