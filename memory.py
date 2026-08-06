"""
Memory System — Persistent knowledge across all sessions.
Facts, preferences, learned commands — auto-extracted from conversations by LLM.
"""

import json
import logging

logger = logging.getLogger("cortex.memory")

EXTRACT_PROMPT = """You are analyzing a conversation between a user and Cortex (an AI assistant).
Extract any facts, preferences, or useful information worth remembering long-term.

Return a JSON array (can be empty []):
[
  {"content": "fact to remember", "type": "fact|preference|learned_command|project_info", "importance": 1-5}
]

Types:
- fact: something true about the user or their environment
- preference: how the user likes things done
- learned_command: a command or workflow the user uses
- project_info: specific info about a project

Importance:
- 5: critical (credentials format, always-use patterns, key project facts)
- 4: very useful (preferences, deployment targets, team conventions)
- 3: useful (tools used, common commands)
- 2: minor (casual mentions)
- 1: probably noise

Only extract things worth remembering across sessions. Return [] if nothing is noteworthy.
Do not extract things already obvious (like the user's username — that's from system context)."""


class MemoryManager:
    """Manages Cortex's persistent memory with LLM-based extraction."""

    def __init__(self, db, llm, config: dict):
        self.db = db
        self.llm = llm
        self.max_facts = config.get("max_facts", 100)
        self.summarize_every = config.get("summarize_every", 10)
        self._turn_counts = {}  # session_id → turn count

    def store_fact(self, content: str, mem_type: str = "fact",
                   source: str = None, importance: int = 3):
        """Store a memory fact directly."""
        self.db.save_memory(type=mem_type, content=content,
                            source=source, importance=importance)
        logger.info(f"Memory stored [{mem_type}, imp={importance}]: {content[:60]}")

    def get_relevant(self, limit: int = 15) -> list:
        """Get top memories by importance for the system prompt."""
        return self.db.get_memories(limit=limit)

    def on_conversation_turn(self, session_id: str):
        """
        Call this after every conversation turn.
        Triggers fact extraction every N turns.
        """
        self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
        count = self._turn_counts[session_id]

        if count % self.summarize_every == 0:
            logger.info(f"Triggering memory extraction for {session_id} (turn {count})")
            self._extract_facts(session_id)

    def _extract_facts(self, session_id: str):
        """Use LLM to extract memorable facts from recent conversation."""
        history = self.db.get_conversations(session_id=session_id, limit=20)
        if len(history) < 2:
            return

        # Format conversation for extraction
        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in history
        )

        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"Conversation:\n{convo_text}"}
        ]

        try:
            raw = self.llm.complete(messages, json_mode=True)
            # LLM may return {"facts": [...]} or just [...]
            parsed = json.loads(raw)
            facts = parsed if isinstance(parsed, list) else parsed.get("facts", [])

            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                content = fact.get("content", "").strip()
                if not content:
                    continue
                self.store_fact(
                    content=content,
                    mem_type=fact.get("type", "fact"),
                    source=session_id,
                    importance=int(fact.get("importance", 3))
                )

            logger.info(f"Extracted {len(facts)} facts from session {session_id}")
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")

    def prune(self):
        """
        Remove oldest low-importance memories when over the limit.
        Keeps high-importance facts safe.
        """
        with self.db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memory")
                count = cur.fetchone()[0]

                if count > self.max_facts:
                    excess = count - self.max_facts
                    cur.execute("""
                        DELETE FROM memory
                        WHERE id IN (
                            SELECT id FROM memory
                            ORDER BY importance ASC, created_at ASC
                            LIMIT %s
                        )
                    """, (excess,))
                    logger.info(f"Pruned {excess} low-importance memories")

    def format_for_prompt(self, limit: int = 15) -> str:
        """Format memories as a string for the LLM system prompt."""
        memories = self.get_relevant(limit=limit)
        if not memories:
            return "(No memories stored yet)"
        lines = []
        for m in memories:
            lines.append(f"- [{m['type']}] {m['content']}")
        return "\n".join(lines)
