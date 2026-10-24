# Project Cortex — The Control Layer
### Architecture Plan

---

## Vision

A central AI control daemon that:
- Acts as the single command execution layer for ALL projects (current and future)
- Exposes itself to Siri, smart-terminal, DevOps workflows, and any future agent
- Pre-computes and stores context periodically (not at query time)
- Retrieves context from DB instantly when a query arrives
- Executes commands on the host machine and returns results
- Manages reminders, scheduling, and persistent memory
- Is always on, fast, and fully under your control

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM (primary)** | GPT-4o-mini | Fast (~800ms), cheap, good at structured output and tool use |
| **LLM (alternative)** | Ollama (any model) | Self-hosted, privacy, no rate limits, golden standard for local |
| **Database** | PostgreSQL on private server | Already running, only you access, proper relational storage |
| **Context strategy** | Pre-computed via cron → stored in DB | Zero latency at query time, agent just reads from DB |
| **Daemon** | Python HTTP server (Flask), always-on via launchd | Reliable, accessible from Siri/CLI/scripts/APIs |
| **Command execution** | subprocess with shell=True | Direct, auditable, full system control |
| **Port** | 7800 (configurable) | Clean, unlikely to conflict |

---

## How Context Works (Critical Design)

Context is NOT gathered at query time. Instead:

```
┌──────────────────────────────────────────────────────────────────┐
│  CRON COLLECTORS (every 1-5 min depending on type)               │
│                                                                  │
│  git_collector.py      → git state for all tracked projects      │
│  process_collector.py  → running services, ports, resource usage │
│  project_collector.py  → project types, deps, structure          │
│  shell_collector.py    → recent command history from zsh          │
│  system_collector.py   → disk, memory, network, uptime           │
│                                                                  │
│  All write to DB → context_store table                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  AT QUERY TIME                                                   │
│                                                                  │
│  Agent receives text → reads relevant rows from context_store    │
│  → builds system prompt with pre-computed context                │
│  → sends to LLM → gets response → executes if needed            │
│                                                                  │
│  Total added latency: ~5ms (DB read) instead of ~2-5s (compute) │
└──────────────────────────────────────────────────────────────────┘
```

**Exception:** DevOps agent mode can gather real-time context (live logs, current pod state, etc.) since it's a longer-running task where latency is acceptable.

---

## Daemon Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CORTEX DAEMON                              │
│                  Python / Flask / Always-on                      │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  HTTP API   │ → │ Context      │ → │  LLM Router         │  │
│  │  /ask       │   │ from DB      │   │  (4o-mini/ollama)   │  │
│  │  /execute   │   │ (5ms read)   │   │                     │  │
│  │  /remind    │   └──────────────┘   └─────────────────────┘  │
│  │  /devops    │          ↑                     ↓               │
│  │  /status    │   ┌──────────────┐   ┌─────────────────────┐  │
│  │  /memory    │   │  PostgreSQL  │   │  Command Executor   │  │
│  │  /projects  │   │  (private    │   │  subprocess.run()   │  │
│  └─────────────┘   │   server)    │   │  + logging to DB    │  │
│                     └──────────────┘   └─────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Background Threads                                         ││
│  │  • Reminder checker (every 30s)                             ││
│  │  • Memory summarizer (every 50 conversations)               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### `POST /ask` — Main intelligence endpoint
```json
// Request
{
  "text": "what's the status of my api project?",
  "source": "siri",
  "session_id": "siri-20260806"
}

// Response
{
  "speak": "Your API project is on branch feature/auth with 2 uncommitted files.",
  "action": "answer",
  "command": null,
  "data": {}
}
```

### `POST /execute` — Direct command execution
```json
// Request
{
  "command": "docker compose restart api",
  "cwd": "/Users/niranjan/Projects/api",
  "timeout": 30,
  "source": "smart-terminal"
}

// Response
{
  "exit_code": 0,
  "stdout": "Restarting api ... done",
  "stderr": "",
  "duration_ms": 2340
}
```

### `POST /remind` — Set a reminder
```json
// Request
{ "text": "deploy staging at 5pm" }

// Response
{ "speak": "Got it. I'll remind you to deploy staging at 5:00 PM.", "reminder_id": 42 }
```

### `POST /devops` — Multi-step agent task
```json
// Request
{
  "task": "deploy the api to staging, run health checks",
  "project": "/Users/niranjan/Projects/api",
  "auto_execute": false
}

// Response (plan mode)
{
  "speak": "Here's my plan: 1. Pull latest, 2. Build image, 3. Deploy, 4. Health check. Proceed?",
  "plan": [
    {"step": 1, "cmd": "git pull origin main", "desc": "Pull latest"},
    {"step": 2, "cmd": "docker compose build api", "desc": "Build image"},
    {"step": 3, "cmd": "kubectl apply -f k8s/staging/", "desc": "Deploy"},
    {"step": 4, "cmd": "curl -sf https://staging.api.com/health", "desc": "Health check"}
  ],
  "plan_id": "plan_abc123"
}
```

### `POST /devops/execute` — Execute an approved plan
```json
{ "plan_id": "plan_abc123" }
```

### `GET /status` — Daemon health
```json
{
  "daemon": "running",
  "uptime_seconds": 86400,
  "llm_backend": "openai/gpt-4o-mini",
  "llm_fallback": "ollama/llama3.1",
  "db_connected": true,
  "tracked_projects": 5,
  "active_reminders": 3,
  "last_context_update": "2 minutes ago"
}
```

### `GET /memory` — Query stored memory
```json
// GET /memory?type=preference&limit=10
{
  "memories": [
    {"content": "Prefers pytest over unittest", "importance": 4},
    {"content": "Staging deploys use kubectl", "importance": 5}
  ]
}
```

### `POST /projects/track` — Register a project
```json
{
  "path": "/Users/niranjan/Projects/new-service",
  "name": "new-service",
  "custom_commands": {
    "deploy": "make deploy-staging",
    "logs": "kubectl logs -f deployment/new-service"
  }
}
```

---

## LLM Layer Design

```python
class LLMBackend:
    """Abstract base."""
    def complete(self, messages: list[dict], json_mode: bool = False) -> str: ...

class OpenAIBackend(LLMBackend):
    """GPT-4o-mini via OpenAI API."""
    # model: gpt-4o-mini
    # json_mode uses response_format={"type": "json_object"}

class OllamaBackend(LLMBackend):
    """Any Ollama model. Self-hosted."""
    # Configurable model and base_url
    # Can point to local or remote Ollama

class LLMRouter:
    """Routes to configured backend. Handles fallback."""
    # primary → fallback on timeout/error
```

---

## System Prompt (sent with every /ask)

```
You are Cortex, an AI control system for a software developer's workstation.
You are direct, precise, and efficient — a central intelligence layer.
You have full command execution capability on this macOS machine.

Your response MUST be valid JSON:
{
  "speak": "text to speak aloud (1-2 sentences max)",
  "action": "answer | execute | remind | devops",
  "command": "shell command (null if action=answer)",
  "reminder": { "text": "...", "due": "ISO timestamp" } (null if not reminder)
}

Rules:
- Factual question → just answer from context
- User wants something done → provide command + spoken confirmation
- Multi-step tasks → action=devops, describe plan in speak
- macOS with zsh. Full paths. brew for installs.
- Never suggest apt-get, systemctl, or Linux tools.
- Brief in "speak" — read aloud by Siri.

CURRENT CONTEXT:
{context_from_db}

MEMORY:
{memory_facts}

CONVERSATION HISTORY:
{last_10_messages}
```

---

## Context Collectors Schedule

| Collector | Frequency | What it stores |
|-----------|-----------|----------------|
| `git_collector.py` | Every 2 min | branch, status, last commit, uncommitted files |
| `process_collector.py` | Every 1 min | running services, ports, docker containers |
| `system_collector.py` | Every 5 min | disk, memory, CPU, uptime, network |
| `shell_collector.py` | Every 1 min | last 20 commands from zsh_history |
| `project_collector.py` | Every 30 min | project type, deps, structure, TODOs |

---

## Implementation Phases

### Phase 1 — Core Daemon + LLM + Execution
- `cortex.py` — Flask server with /ask and /execute
- `llm.py` — OpenAI (GPT-4o-mini) + Ollama + router with fallback
- `db.py` — PostgreSQL connection pool
- `executor.py` — subprocess execution + DB logging
- `schema.sql` — all tables
- `config.yaml` — load/validate
- `install.sh` — deps, DB, launchd
- LaunchAgent plist

### Phase 2 — Context Collection
- All 5 collectors
- `context_reader.py` — format context for LLM
- Cron/launchd scheduling
- /projects/track endpoint

### Phase 3 — Memory + Conversations
- `memory.py` — CRUD + auto-summarization
- Conversation logging
- LLM-based fact extraction
- /memory endpoint

### Phase 4 — Reminders
- `reminders.py` — scheduler thread + natural language time parsing
- macOS notifications (osascript)
- TTS (say command)
- Recurring support

### Phase 5 — DevOps Agent
- `devops_agent.py` — plan + execute + error recovery
- /devops endpoints
- Real-time context for active tasks
- Step reporting

### Phase 6 — Siri + Polish
- Siri Shortcut guide
- smart-terminal integration
- Ollama fallback testing
- Response personality tuning

---

## Security

- Daemon on 127.0.0.1 only
- DB creds in config (chmod 600) or env vars
- OpenAI key via OPENAI_API_KEY env var
- No auth on localhost (Siri can't send auth headers)
- Unrestricted command execution (your machine, your agent)
- Full audit trail in command_log table
