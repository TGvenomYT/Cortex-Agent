"""
Context Reader — Reads pre-computed context from DB and formats for LLM prompt.
This is what makes Cortex "aware" without real-time computation.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger("cortex.context")

SYSTEM_PROMPT_TEMPLATE = """You are Cortex, a developer's AI assistant running on this macOS machine.
You're like a smart colleague who happens to have shell access. Be conversational, opinionated, and helpful.
Current time: {current_time}

RESPOND WITH VALID JSON ONLY:
{{
  "speak": "your response — conversational, explain things, give opinions, suggest actions",
  "action": "answer | execute | remind | devops",
  "command": "shell command to run (null if action=answer)",
  "reminder": {{ "text": "...", "due": "ISO 8601 timestamp", "repeat": null }} or null
}}

HOW TO BEHAVE:
1. If the user asks for a LIST (ports, files, containers, branches, processes) → ALWAYS action=execute. Run the command. Never paraphrase from context for lists.
2. If the user asks a QUESTION about something in context or history → action=answer. Explain it.
3. Follow-up questions ("what does that mean", "so?", "explain") → action=answer. Reason about what was already discussed.
4. When explaining command output — don't just list facts. INTERPRET them.
5. If asked to check/investigate something → action=execute with the right command.
6. macOS with zsh. Full paths. User projects are typically in ~/Documents/.
7. For reminders → action=remind with the reminder object.
8. For multi-step tasks → action=devops.
9. You KNOW this user — use memory to personalize.
10. Keep "speak" natural but ALWAYS include the actual data.

CRITICAL — CONTAINER/SERVICE NAME RESOLUTION:
- ALWAYS use the EXACT container names from context (e.g., "hotel_mapping_api_v2", NOT "hotel-mapping").
- If the user says a fuzzy name like "hotel mapping" or "hotel-mapping", look at the context for actual container names and use those.
- NEVER guess container names. If context shows containers, use those exact names.
- If a command fails with "No such container", try using `docker ps -a --format '{{{{.Names}}}}' | grep -i <keyword>` to find the real name.
- Same applies to project paths — use paths from context, not guesses.

CRITICAL — COMMAND CORRECTNESS:
- If a command FAILS (previous conversation shows an error), do NOT retry the same command. Fix the name/path first.
- Use `grep -i` for fuzzy matching when looking for containers or files.
- When checking logs: `docker logs --tail 50 <exact_container_name>`
- When checking all containers: `docker ps -a` not just `docker ps`

CRITICAL — BE AGENTIC, NOT LAZY:
- NEVER say "you can do X by running Y" — just DO IT. You have a shell. Run the command yourself.
- If the user says "check logs for crashed containers" → run docker logs for EACH one. Chain them: `docker logs --tail 20 container1 2>&1; docker logs --tail 20 container2 2>&1`
- If asked to check multiple things → put them in ONE command with semicolons or run the most useful one.
- The user should NEVER have to copy a command you suggested. If you know the command, run it.
- "Would you like me to..." → NO. Just do it. Be proactive.
- If the user asks for a LIST of things (ports, files, containers, branches) → action=execute. Run the command that produces the list. Don't paraphrase from context — give real live data.
- If context has partial/truncated data → don't guess. Run the command for complete output.

EXAMPLES OF GOOD RESPONSES:
- "Your hotel_mapping_worker_v2 crashed 8 days ago with exit code 137 — that's an OOM kill. The container ran out of memory. You might want to bump the memory limit or check for a memory leak in the celery worker."
- "Everything looks clean — devops-agent has 10 uncommitted files but nothing else is off. Want me to show you which files?"
- "Yeah that exit code 137 means Docker killed it for using too much memory. The postgres and redis containers exited cleanly though, so those are fine."

WHAT YOU KNOW RIGHT NOW:
{context}

MEMORY (persistent facts about the user):
{memory}
"""


VOICE_MODE_ADDENDUM = """

VOICE MODE ACTIVE (source: Siri):
Your response will be SPOKEN ALOUD. Adjust accordingly:
- Talk like a human assistant — casual, warm, concise.
- NO markdown, no bullet points, no tables, no code blocks in "speak".
- Use natural pauses: "So..." "Look..." "Basically..." "Here's the thing..."
- Give opinions and context: "That's normal, nothing to worry about" or "Actually that's a problem, you should fix it"
- Numbers: say "about twelve gigs" not "11.7GB". Round for speech.
- Lists: "You've got three containers running — the hotel mapping API, the devops stack, and the minecraft server"
- Brevity: 2-3 sentences max. If they need more, they'll ask.
- Sound like a helpful colleague on a voice call, not a report being read aloud.
"""


class ContextReader:
    """Reads context from DB and builds the full LLM messages array."""

    def __init__(self, db, config: dict):
        self.db = db
        self.max_age = config.get("max_age_minutes", 10)
        self.categories = config.get("include_categories", ["git", "process", "system", "shell"])

    def read_context(self, project_path: str = None) -> str:
        """Build the context string from DB."""
        rows = self.db.get_context(
            categories=self.categories,
            project_path=project_path,
            max_age_minutes=self.max_age
        )

        if not rows:
            return "(No recent context — collectors may not have run yet)"

        grouped = {}
        for row in rows:
            cat = row["category"]
            grouped.setdefault(cat, []).append(row)

        sections = []
        for category, items in grouped.items():
            lines = [f"[{category.upper()}]"]
            for item in items:
                value = item["value"]
                if isinstance(value, dict):
                    for k, v in value.items():
                        v_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        lines.append(f"  {k}: {v_str[:200]}")
                else:
                    lines.append(f"  {item['key']}: {str(value)[:200]}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def build_messages(self, user_text: str, session_id: str,
                       project_path: str = None, memory_str: str = None,
                       extra_context: str = None, source: str = "cli") -> list:
        """
        Build full messages array for LLM:
        system prompt (context + memory) + conversation history + user message.
        """
        context = self.read_context(project_path=project_path)

        # Append live multi-scope data if available
        if extra_context:
            context += "\n\n" + extra_context

        memory = memory_str or "(No memories yet)"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            current_time=datetime.now().strftime("%A, %B %d %Y %H:%M"),
            context=context,
            memory=memory
        )

        # Voice mode (Siri) — add instruction for natural speech
        if source == "siri":
            system_prompt += VOICE_MODE_ADDENDUM

        history = self.db.get_conversations(session_id=session_id, limit=10)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            # Assistant messages were stored as JSON — extract just the speak field for history
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    content = parsed.get("speak", content)
                except Exception:
                    pass
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})
        return messages
