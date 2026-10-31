"""
Memory System — what Cortex knows about its owner, kept across every chat.

After each message, what the person said about themselves is read and saved: new facts
added, corrections replacing what they correct, repeats ignored. Only their own words are
read, never replies, web pages or tool output, so nothing outside can plant a memory.

    rules       standing instructions ("always call me Boss"), in every prompt, never pruned
    core        the few most important facts, in every prompt
    relevant    whatever bears on the current question, found by meaning, added for that turn

Rules and core sit in the cached part of the prompt and rarely change. Relevant memories
go in the newest message, so looking them up doesn't throw the model's prompt cache away.

Passwords, keys and card numbers are never kept: they'd go into prompts from then on, and
to the provider too if a cloud model is in use.
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from cortex.core import safety
from cortex.llm.schemas import FACTS_SCHEMA

logger = logging.getLogger("cortex.memory")

EXTRACT_PROMPT = """You keep notes about a person, from what they say. You're given their own words, and the notes you already hold.

Return JSON: {{"facts": [{{"content": "...", "type": "...", "importance": 1-5, "replaces": null}}]}}

Types:
- fact: something true about them, their work or their life
- preference: how they like things done
- project_info: about a project of theirs
- learned_command: a command or workflow they use
- rule: a standing instruction for how to behave, from now on ("always...", "never...", "call me...")

Importance: 5 who they are and standing instructions, 4 useful preferences and work details, 3 useful, 2 minor, 1 noise.

"replaces": if this corrects or updates a note below (they moved, changed job, changed their mind), give that note's number, so the old one is replaced. Otherwise null.

Write each note as one short standalone sentence about "the user". Only keep what would still matter in another conversation: not this task's details, not questions, not requests. Return {{"facts": []}} if there's nothing. Never note passwords, keys, tokens or card numbers.

NOTES YOU ALREADY HOLD:
{known}"""

EXTRACTABLE_TYPES = {"fact", "preference", "learned_command", "project_info", "rule"}
_RULE_WORDS = re.compile(r"\b(always|never|from now on|call me|don'?t ever|stop \w+ing|whenever|every time)\b", re.I)
_SELF = re.compile(r"\b(i|i'm|i am|i've|i'd|i'll|my|mine|we|we're|our|me)\b", re.I)
_REQUEST = re.compile(r"^(please\s+|can you\s+|could you\s+)?(show|list|run|check|find|open|make|create|write|plot|draw|"
                      r"summari[sz]e|search|tell|give|what|how|why|who|where|when|which|is|are|do|does|did|fix|update|"
                      r"delete|remove|install|send|read|explain|help)\b", re.I)
_ABOUT_ME = re.compile(r"\b(about me|know about|remember about|what do you remember|who am i)\b", re.I)
MIN_SIMILARITY = 0.45
SAME_NOTE = 0.8


def worth_learning(text: str) -> bool:
    """Could this message hold something to remember? Spares a model call for plain requests."""
    t = (text or "").strip()
    if len(t) < 8 or t.startswith("["):
        return False
    if _RULE_WORDS.search(t):
        return True
    for sentence in re.split(r"(?<=[.!?\n])\s+", t):
        s = sentence.strip()
        if s and not s.endswith("?") and _SELF.search(s) and not _REQUEST.match(s):
            return True
    return False


class MemoryManager:
    """Keeps what Cortex knows about its owner, learned from what they say."""

    def __init__(self, db, llm, config: dict):
        self.db = db
        self.llm = llm
        self.max_facts = config.get("max_facts", 100)
        # Learning makes a model call, so it runs off to the side of the reply.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-learn")
        self._timers: dict = {}

    # ── storing ───────────────────────────────────────────────────────────
    @staticmethod
    def _vector(text: str):
        from cortex.llm.embeddings import embed, to_pgvector
        vec = embed(text)
        return to_pgvector(vec) if vec else None

    def store_fact(self, content: str, mem_type: str = "fact", source: str = None, importance: int = 3) -> dict:
        """Save one memory. -> {"status": "stored"|"known"|"secret", "id": ...}"""
        content = (content or "").strip()
        if safety.contains_secret(content):
            logger.info("refused to store something that looks like a secret")
            return {"status": "secret", "id": None}
        same = self._same_note(content)
        if same:
            return {"status": "known", "id": same["id"]}
        mem_id = self.db.save_memory(type=mem_type, content=content, source=source, importance=importance,
                                     embedding=self._vector(content))
        logger.info(f"Memory stored [{mem_type}, imp={importance}]: {content[:60]}")
        return {"status": "stored", "id": mem_id}

    def _same_note(self, content: str) -> dict | None:
        words = self.db._keywords(content)
        if not words:
            return None
        numbers = re.findall(r"\d+", content)
        for m in self.db.get_memories(limit=2000):
            other = self.db._keywords(m["content"])
            # "ticket 14" and "ticket 21" read alike, but they're different facts
            if other and len(words & other) / len(words | other) >= SAME_NOTE \
                    and numbers == re.findall(r"\d+", m["content"]):
                return m
        return None

    def remember_rule(self, content: str, source: str = "conversation") -> dict:
        """Store a standing behavioral instruction — always obeyed, never pruned, never capped."""
        return self.store_fact(content, mem_type="rule", source=source, importance=5)

    def forget(self, memory_id: int = None, search: str = None) -> str:
        """Delete a memory by id, or by a content snippet if the id isn't known."""
        if memory_id:
            return f"forgot memory #{memory_id}" if self.db.delete_memory(memory_id) \
                else f"no memory #{memory_id} found"
        if search:
            hits = [m for m in self.db.get_memories(limit=500)
                    if search.lower() in m["content"].lower()]
            if not hits:
                return f"couldn't find a stored memory matching '{search}'"
            self.db.delete_memory(hits[0]["id"])
            return f"forgot: {hits[0]['content'][:80]}"
        return "need either a memory id or a search snippet to forget something"

    # ── learning from a chat ──────────────────────────────────────────────
    def on_conversation_turn(self, session_id: str, delay: float = 0):
        """Call after every turn. With a delay it waits for a pause in the chat: the model only
        caches one prompt at a time, so learning mid-conversation would slow the next reply."""
        def start():
            self._executor.submit(self.learn, session_id)
        old = self._timers.pop(session_id, None)
        if old:
            old.cancel()
        if not delay:
            start()
            return
        timer = threading.Timer(delay, start)
        timer.daemon = True
        self._timers[session_id] = timer
        timer.start()

    def learn(self, session_id: str) -> int:
        """Read what's new in a chat. If the model can't be reached nothing is marked as read,
        so it's picked up next time. -> how many notes changed."""
        rows = self.db.recent_messages(session_id, after_id=self.db.get_learned_upto(session_id), limit=40)
        if not rows:
            return 0
        said = [r["content"] for r in rows if r["role"] == "user" and worth_learning(r["content"])]
        if not said:
            self.db.set_learned_upto(session_id, rows[-1]["id"])
            return 0
        try:
            import cortex.util.power as power
            if power.is_constrained()[0]:
                logger.info("learning postponed: machine is busy")
                return 0
        except Exception:
            pass
        text = "\n".join(said)[:6000]
        known = self._nearest(text, 12)
        listing = "\n".join(f"{i + 1}. {m['content']}" for i, m in enumerate(known)) or "(none yet)"
        try:
            raw = self.llm.complete([{"role": "system", "content": EXTRACT_PROMPT.format(known=listing)},
                                     {"role": "user", "content": f"What they said:\n{text}"}],
                                    json_mode=True, schema=FACTS_SCHEMA)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning(f"learning failed, will retry: {e}")
            return 0
        facts = parsed if isinstance(parsed, list) else parsed.get("facts", [])
        changed = sum(self._take(f, known, text, session_id) for f in facts if isinstance(f, dict))
        self.db.set_learned_upto(session_id, rows[-1]["id"])
        logger.info(f"Learned {changed} notes from {session_id}")
        if changed:
            self.prune()
        return changed

    def _take(self, fact: dict, known: list, said: str, session_id: str) -> int:
        content = (fact.get("content") or "").strip()
        if not content:
            return 0
        kind = fact.get("type") if fact.get("type") in EXTRACTABLE_TYPES else "fact"
        if kind == "rule" and not _RULE_WORDS.search(said):
            kind = "preference"     # obeyed on every turn, so it needs an explicit instruction
        try:
            old = int(fact.get("replaces") or 0)
        except (TypeError, ValueError):
            old = 0
        if 1 <= old <= len(known):
            if safety.contains_secret(content):
                return 0
            return 1 if self.db.update_memory(known[old - 1]["id"], content, self._vector(content)) else 0
        importance = 5 if kind == "rule" else max(1, min(4, int(fact.get("importance") or 3)))
        return 1 if self.store_fact(content, mem_type=kind, source=session_id, importance=importance)["status"] == "stored" else 0

    def prune(self):
        """
        Remove oldest low-importance memories when over the limit.
        Keeps high-importance facts safe, and never touches rules — a standing
        instruction doesn't get to quietly age out just because the fact table
        filled up.
        """
        with self.db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memory WHERE type != 'rule'")
                count = cur.fetchone()[0]

                if count > self.max_facts:
                    excess = count - self.max_facts
                    cur.execute("""
                        DELETE FROM memory
                        WHERE id IN (
                            SELECT id FROM memory
                            WHERE type != 'rule'
                            ORDER BY importance ASC, created_at ASC
                            LIMIT %s
                        )
                    """, (excess,))
                    logger.info(f"Pruned {excess} low-importance memories")

    # ── recalling ─────────────────────────────────────────────────────────
    def get_relevant(self, limit: int = 15) -> list:
        """Top memories by importance."""
        return self.db.get_memories(limit=limit)

    def _nearest(self, text: str, limit: int) -> list:
        """Memories closest to `text`, by meaning where the embedder is up, and by shared words."""
        from cortex.llm.embeddings import embed
        hits, seen = [], set()
        vec = embed(text[:2000])
        by_meaning = [m for m in self.db.search_memories(vec, limit) if m["similarity"] >= MIN_SIMILARITY] if vec else []
        for m in by_meaning + self.db.search_memories_keyword(text, limit):
            if m["id"] not in seen:
                seen.add(m["id"])
                hits.append(m)
        return hits[:limit]

    def format_core(self, limit: int = 8) -> str:
        """Rules and the most important facts: the part of memory in every prompt. Ordered by
        age, so a new note doesn't reshuffle the prompt and spoil the cache."""
        rules = self.db.get_memories(type="rule", limit=200)
        core = [m for m in self.db.get_memories(limit=limit + len(rules))
                if m["type"] != "rule" and m["importance"] >= 5][:limit]
        parts = []
        if rules:
            parts.append(
                "STANDING RULES THE USER GAVE YOU — these are not suggestions, obey them "
                "every time even if it feels repetitive or no one reminds you again:\n" +
                "\n".join(f"- {r['content']}" for r in sorted(rules, key=lambda m: m["id"])))
        if core:
            parts.append("ABOUT THEM:\n" + "\n".join(f"- {m['content']}" for m in sorted(core, key=lambda m: m["id"])))
        return "\n\n".join(parts) or "(Nothing yet: you'll learn about them as you talk.)"

    def format_relevant(self, query: str, limit: int = 8) -> str:
        """What else bears on this question. '' if nothing does."""
        if _ABOUT_ME.search(query or ""):
            hits = self.db.get_memories(limit=40)
        else:
            hits = self._nearest(query or "", limit)
        hits = [m for m in hits if m["type"] != "rule" and m["importance"] < 5]
        return "\n".join(f"- {m['content']}" for m in hits)

    def format_for_prompt(self, limit: int = 15) -> str:
        """Everything worth knowing with no particular question in mind: background jobs."""
        rules = self.db.get_memories(type="rule", limit=200)
        facts = [m for m in self.get_relevant(limit=limit + len(rules)) if m["type"] != "rule"][:limit]
        if not rules and not facts:
            return "(No memories stored yet)"

        parts = []
        if rules:
            parts.append(
                "STANDING RULES THE USER GAVE YOU — these are not suggestions, obey them "
                "every time even if it feels repetitive or no one reminds you again:\n" +
                "\n".join(f"- {r['content']}" for r in rules)
            )
        if facts:
            parts.append(
                "OTHER REMEMBERED FACTS / PREFERENCES:\n" +
                "\n".join(f"- [{m['type']}] {m['content']}" for m in facts)
            )
        return "\n\n".join(parts)
