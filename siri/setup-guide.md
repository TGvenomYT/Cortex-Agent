# Cortex — Siri Integration Guide

Cortex runs on `http://localhost:7800`. Siri Shortcuts call it over localhost.
All shortcuts work on your Mac via "Hey Siri" when Cortex daemon is running.

---

## Prerequisites

- Cortex daemon running: `CORTEX_DB_PASSWORD=... OPENAI_API_KEY=... .venv/bin/python3 cortex.py`
- Verify: `curl http://localhost:7800/status`
- Shortcuts app on macOS

---

## Shortcut 1: "Hey Siri, Ask Cortex" — Main shortcut

This is the core shortcut. You speak, Cortex responds, Siri reads it back.

**Actions:**

1. **Dictate Text**
   - Prompt: *(leave empty — Siri will listen)*
   - Store result in variable: `input`

2. **Get Contents of URL**
   - URL: `http://localhost:7800/ask`
   - Method: `POST`
   - Request Body: `JSON`
     ```json
     {
       "text": [input variable],
       "source": "siri",
       "session_id": "siri-daily"
     }
     ```
   - Headers: `Content-Type: application/json`
   - Store result in variable: `response`

3. **Get Dictionary Value**
   - Get value for key: `speak`
   - From: `response`
   - Store result in variable: `reply`

4. **Speak Text**
   - Text: `reply`

**Siri phrase:** "Ask Cortex" or "Hey Cortex"

---

## Shortcut 2: "Hey Siri, Remind Me" — Reminders

1. **Dictate Text** → variable `input`

2. **Get Contents of URL**
   - URL: `http://localhost:7800/remind`
   - Method: `POST`
   - Body JSON:
     ```json
     { "text": [input] }
     ```

3. **Get Dictionary Value**: key `speak` from response

4. **Speak Text**: the speak value

**Siri phrase:** "Remind Me" or "Set a Cortex reminder"

---

## Shortcut 3: "Hey Siri, Cortex Status" — Quick status

1. **Get Contents of URL**
   - URL: `http://localhost:7800/ask`
   - Method: `POST`
   - Body JSON:
     ```json
     {
       "text": "give me a quick status update — system health, what's running, any important git changes",
       "source": "siri",
       "session_id": "siri-daily"
     }
     ```

2. **Get Dictionary Value**: key `speak`

3. **Speak Text**

**Siri phrase:** "Cortex Status"

---

## Shortcut 4: "Hey Siri, Run Command" — Execute anything

1. **Dictate Text** → variable `cmd`

2. **Get Contents of URL**
   - URL: `http://localhost:7800/ask`
   - Method: `POST`
   - Body JSON:
     ```json
     {
       "text": [cmd],
       "source": "siri",
       "session_id": "siri-daily"
     }
     ```
   *(The /ask endpoint figures out intent — if it's a command, it runs it automatically)*

3. **Get Dictionary Value**: key `speak`

4. **Speak Text**

**Siri phrase:** "Run Command" or "Cortex Run"

---

## Shortcut 5: "Hey Siri, DevOps" — Plan a task

1. **Dictate Text** → variable `task`

2. **Get Contents of URL**
   - URL: `http://localhost:7800/devops`
   - Method: `POST`
   - Body JSON:
     ```json
     {
       "task": [task],
       "auto_execute": false
     }
     ```
   - Store result → variable `plan_response`

3. **Get Dictionary Value**: key `speak` from `plan_response` → `plan_summary`

4. **Speak Text**: `plan_summary`

5. **Choose from Menu**
   - Prompt: "Execute this plan?"
   - Options: `Execute`, `Cancel`

6. **If Execute:**
   - Get Dictionary Value: key `plan_id` from `plan_response` → `pid`
   - Get Contents of URL:
     - URL: `http://localhost:7800/devops/execute`
     - Method: POST
     - Body: `{"plan_id": [pid]}`
   - Get Dictionary Value: key `speak`
   - Speak Text

**Siri phrase:** "DevOps" or "Cortex Deploy"

---

## Tips

- The `session_id: "siri-daily"` keeps your Siri conversations in one thread per day, giving Cortex memory of earlier interactions in the same day.
- Change `session_id` to `"siri-persistent"` if you want a single rolling thread across all days.
- The `/ask` endpoint is smart — you don't need separate shortcuts for every action. Saying "remind me to deploy at 5pm" through the main Ask Cortex shortcut will create a reminder automatically.
- Cortex already knows your name (Niranjan), your preferences (pytest, docker compose, staging-first deploys) from memory.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Shortcut fails silently | Check daemon is running: `curl localhost:7800/status` |
| Siri says "I couldn't connect" | Localhost shortcuts only work on the same Mac — not iPhone |
| Response is empty | Check OPENAI_API_KEY is set in daemon environment |
| Reminder doesn't fire | Daemon must be running; check logs in `logs/` folder |
