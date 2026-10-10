# Cortex — Siri Setup

You only need **one shortcut**. Cortex figures out what you want — questions, commands, reminders, deployments — all through natural speech.

---

## The One Shortcut You Need

**Name it:** "Cortex" (or "Hey Cortex")

When you say **"Hey Siri, Cortex"** — Siri listens, sends your speech to Cortex, and speaks back the response. That's it.

### How to create it:

1. Open **Shortcuts** app on your Mac
2. Tap **+** to create a new shortcut
3. Name it **"Cortex"**
4. Add these actions in order:

---

**Action 1:** Dictate Text
- Leave the prompt empty — Siri will just listen

**Action 2:** Get Contents of URL
- URL: `http://localhost:7800/ask`
- Method: POST  
- Headers: Content-Type → application/json
- Request Body → JSON:
  - text: *(Dictated Text variable)*
  - source: `siri`
  - session_id: `siri`

**Action 3:** Get Dictionary Value
- Key: `speak`

**Action 4:** Speak Text
- *(the value from Action 3)*

---

That's it. Now you can say:

- "Hey Siri, Cortex" → *"how's my server?"*
- "Hey Siri, Cortex" → *"remind me to deploy staging at 5"*
- "Hey Siri, Cortex" → *"any containers down?"*
- "Hey Siri, Cortex" → *"what's using port 3000?"*

Cortex handles the intent — reminders, commands, status checks, deployments — all from one phrase.

---

## How it sounds

**You:** "Hey Siri, Cortex"  
**Siri:** *(listens)*  
**You:** "Any problems with my docker containers?"  
**Cortex (via Siri):** "Everything's running fine. You've got six containers up — hotel mapping, devops stack, and the minecraft server. No crashes."

**You:** "Hey Siri, Cortex"  
**Siri:** *(listens)*  
**You:** "Remind me to check logs at 2pm"  
**Cortex (via Siri):** "Got it, I'll remind you to check logs at 2."

---

## Why one shortcut works

The `/ask` endpoint with `source: "siri"` does three things differently:

1. **Voice-optimized responses** — speaks in natural sentences, no jargon, rounds numbers ("about 4 gigs" not "3.6GB")
2. **Intent detection** — automatically routes to reminders, commands, devops based on what you said
3. **Conversation memory** — uses `session_id: "siri"` so Cortex remembers what you talked about earlier in the day

---

## Prerequisites

- Cortex daemon running (starts automatically after install)
- Same Mac where Siri is active (localhost only)
- Check it's up: open terminal → `curl localhost:7800/status`

---

## Optional: Shorter Trigger

If you want to skip saying "Cortex" separately, you can name the shortcut something more natural like:

- **"Check my server"** — for status queries
- **"Dev update"** — for project/git status

But honestly the single "Cortex" shortcut handles everything. Less to maintain.

---

## Troubleshooting

- **Siri says nothing back** → daemon isn't running. Check `launchctl list | grep cortex`
- **Siri says "I couldn't connect"** → only works on the Mac, not iPhone (localhost)
- **Response is generic** → check OPENAI_API_KEY is set in daemon environment
- **Reminder didn't fire** → daemon needs to be running continuously for the reminder thread to work
