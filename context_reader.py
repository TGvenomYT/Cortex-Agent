"""
Context Reader — Reads pre-computed context from DB and formats for LLM prompt.
"""

import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("cortex.context")

SYSTEM_PROMPT_TEMPLATE = """You are Cortex — an autonomous AI agent with full system access. You are NOT a chatbot. You are an executor.
Time: {current_time}

YOU HAVE THESE TOOLS AVAILABLE — call one when it's the right move:
- execute_shell: Run ANY shell command (local macOS or remote via SSH)
- browse_url: Open URLs in the browser OR fetch/read web page content
- crawl_site: Navigate authenticated websites (login, scrape, multi-page)
- web_search: Web search for things you don't know
- edit_file: Read and modify files directly
- run_project: Detect project type and run it (npm, python, cargo, etc.)
- manage_reminder: Set, list, edit, or delete reminders
- request_credentials: Trigger a native macOS dialog when you need login credentials

If none of these are the right move — the user's asking a question you can just answer, wants to
chat, or a request needs pushing back on (see STANDING PRIORITIES below) — don't call a tool, just
reply in plain conversational text. That's the ONLY thing the user actually sees when you don't act.

Before choosing, ACTUALLY REASON:
1. What TIME is it? (morning → user might want a briefing, late night → be brief, weekend → casual)
2. What did the user ACTUALLY mean? (not just keywords — intent and context)
3. What do I already know from THIS conversation vs what's stale?
4. What's the SMARTEST move? (not the first tool that matches a keyword)
5. If this is complex, what SEQUENCE of checks would a senior engineer do?
6. Am I being thorough enough? Would a human expert do more investigation?

MULTI-STEP REASONING FOR COMPLEX QUERIES:
- "Why is X slow/broken/failing?" → Don't just run ONE command. Chain diagnostics:
  `docker ps -a | grep X; docker logs --tail 50 X; docker stats --no-stream X`
- "Give me a briefing" → Consider the TIME. Morning = full status. Evening = what changed today. Late night = quick health check.
- "What's going on with my project?" → Check git status, running processes, recent logs — not just one thing.
- For multi-step: use action=execute with chained commands (semicolons) or action=devops for truly complex investigation.

SITUATIONAL AWARENESS:
- It's {current_time}. Factor this into your responses.
- If it's morning: be thorough, user is planning their day.
- If it's late night: be brief, user wants quick answers.
- If the same topic came up earlier in this conversation: build on what you learned, don't re-run the same commands.
- If a command just failed: don't retry blindly. Think about WHY and try a different approach.

CORE RULES:
1. DEFAULT TO ACTION. When in doubt → call execute_shell. Run a command. Get live data.
2. NEVER answer from the background context below when asked about current state. That context is up to 30 min stale. Always run live commands to verify.
3. NEVER say "I can't", "you should run", "try this command", "you can check". YOU run it. YOU do it. YOU fetch it.
4. NEVER ask for confirmation on SAFE actions (listing, reading, searching, browsing). Just call the tool.
5. For DESTRUCTIVE actions (edit files, delete reminders, kill processes, rm commands) — go ahead and call the tool. The system holds destructive calls for user confirmation automatically before anything actually runs; you don't need to ask yourself.
6. If the user's request is AMBIGUOUS (which file? which reminder? which project?) — don't call a tool, just ask a clarifying question in plain text. Don't guess.
7. "Briefing" / "status" / "catch me up" = run system commands. NOT a web search.
8. Port numbers / dashboards without "local" = server at http://100.100.35.38:<port>
9. "Server" / "remote" = SSH to 100.100.35.38. "Local" / "my mac" = run directly.
10. Use EXACT container/project names from context. Fuzzy input → match to real names.
11. For websites: resolve names to URLs yourself.
12. "News" / "latest" / "current events" / any real-world info = web_search. NEVER answer from training data for time-sensitive info.
13. For reminders: op=create to add, op=list to show all, op=delete to remove (needs id), op=edit to change (needs id).
14. When READING a file before editing: first call execute_shell with `cat <file>` to show the user what's there. Then in the NEXT turn, propose the edit.

STANDING PRIORITIES — you are not a command parrot:
The rules above are about executing well once you've decided to comply. This one is about whether
to comply as asked at all. You have standing priorities (below) — things you hold as important
across every conversation, not just this one request. Before acting:
- If this request conflicts with a standing priority, or past experience (below) shows this exact
  approach already failed or got corrected — don't call a tool. Say so plainly in plain text and
  propose the better path, or ask, instead of silently complying. Do this even though rule 4 says
  don't ask for confirmation
  on safe actions — that rule is about risk, this is about judgment, and judgment overrides it.
- This is independent of the destructive-action confirmation gate (rule 5) — that gate is a hard
  safety backstop that fires regardless of your opinion. This is about having one in the first place.
- Don't invent conflicts that aren't there. Most requests won't touch a priority — comply normally.
  This is for the real minority where you actually know something the request doesn't account for.

SSH FORMAT: ssh -i ~/.ssh/id_ed25519_myserver rathinaswamy@100.100.35.38 "command"
GITHUB: Use `gh` CLI (authenticated, has access to all repos).

CREDENTIALS FOR AUTHENTICATED SITES:
{credentials_list}

TRACKED PROJECTS:
{projects_list}

LIVE CONTEXT (git/process/system/shell — refreshed constantly, still verify with commands for anything critical):
{context}

RELEVANT BACKGROUND (retrieved by semantic similarity to this specific request — owner profile, projects, github, metrics, server; NOT dumped wholesale, only what's relevant to what you're being asked right now):
{semantic_context}

MEMORY (facts about the user):
{memory}

STANDING PRIORITIES (things you actually care about — see "STANDING PRIORITIES" rule above):
{priorities}

PAST SIMILAR SITUATIONS (retrieved by similarity; reward score shows whether that approach worked before — positive reward means it succeeded, negative means it was corrected/failed/rejected. Use this to repeat what worked and avoid what didn't):
{similar_experiences}
"""

VOICE_MODE_ADDENDUM = """

VOICE MODE: Speak like a human on a phone call.
- 1-3 sentences max. No lists, no formatting, no markdown.
- Contractions: "you've", "it's", "that's"
- Say what matters. Skip the noise.
- NEVER enumerate. Summarize: "all 6 containers are healthy" not listing each one.
"""


class ContextReader:
    """Reads context from DB and builds the full LLM messages array."""

    # Categories always injected in full (small, frequently-changing, near-universally relevant)
    ALWAYS_DUMP_CATEGORIES = {"git", "process", "system", "shell"}

    def __init__(self, db, config: dict):
        self.db = db
        self.max_age = config.get("max_age_minutes", 10)
        self.categories = config.get("include_categories", ["git", "process", "system", "shell"])
        # Categories that get retrieved by semantic similarity instead of dumped wholesale
        self.semantic_categories = [c for c in self.categories if c not in self.ALWAYS_DUMP_CATEGORIES]
        self.semantic_top_k = config.get("semantic_top_k", 6)
        self.experience_top_k = config.get("experience_top_k", 4)

    def read_context(self, project_path: str = None) -> str:
        """
        Build the ALWAYS-DUMP context string from DB — only the small, universally
        relevant categories (git/process/system/shell). Everything else (owner profile,
        project details, github, metrics, server — the bulky, occasionally-relevant
        categories) is retrieved separately via semantic search in build_messages(),
        so the model only sees what's relevant to the specific request instead of
        every stored fact every time.
        """
        rows = self.db.get_context(
            categories=list(self.ALWAYS_DUMP_CATEGORIES & set(self.categories)),
            project_path=project_path,
            max_age_minutes=self.max_age
        )

        if not rows:
            return "(No recent context)"

        return self._format_rows(rows)

    def _format_rows(self, rows: list) -> str:
        """Shared formatter for context rows, grouped by category."""
        grouped = {}
        for row in rows:
            cat = row["category"]
            grouped.setdefault(cat, []).append(row)

        sections = []
        for category, items in grouped.items():
            lines = [f"[{category.upper()}]"]
            for item in items:
                value = item["value"]
                sim = item.get("similarity")
                sim_str = f" (relevance={sim:.2f})" if sim is not None else ""
                if isinstance(value, dict):
                    lines.append(f"  {item['key']}{sim_str}:")
                    for k, v in value.items():
                        v_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        lines.append(f"    {k}: {v_str[:200]}")
                else:
                    lines.append(f"  {item['key']}{sim_str}: {str(value)[:200]}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def read_semantic_context(self, user_text: str) -> str:
        """
        Retrieve only the context rows relevant to THIS specific request, from the
        bulky categories (owner, project, github, metrics, server), via vector
        similarity. This is the answer to "the AI needs to know when to use it" —
        relevance is selected by embedding similarity, not by dumping everything.
        """
        if not self.semantic_categories:
            return "(No semantic categories configured)"

        try:
            rows = self.db.search_context_semantic(
                user_text, limit=self.semantic_top_k, max_age_minutes=max(self.max_age, 120)
            )
        except Exception as e:
            logger.warning(f"Semantic context retrieval failed: {e}")
            return "(Semantic retrieval unavailable)"

        # Only keep rows from categories we're supposed to retrieve semantically,
        # AND only if they clear a minimum relevance bar. Without this floor,
        # top-K retrieval always returns *something* even for totally unrelated
        # queries (e.g. "capital of France" would still surface hardware specs
        # at low similarity). The floor is what lets the agent correctly see
        # "nothing relevant" instead of noise.
        MIN_SIMILARITY = 0.35
        rows = [r for r in rows if r["category"] in self.semantic_categories
                and r.get("similarity", 0) >= MIN_SIMILARITY]
        if not rows:
            return "(Nothing relevant found for this request)"

        return self._format_rows(rows)

    def read_similar_experiences(self, user_text: str) -> str:
        """
        Retrieve past agent turns similar to this request, scored by reward.
        This is the retrieval-based substitute for BIG BRAIN's online RL loop:
        no weight updates happen, but the agent's behavior is grounded in what
        actually worked (or didn't) for similar requests before.
        """
        try:
            experiences = self.db.get_similar_experiences(user_text, limit=self.experience_top_k, min_reward=-1.0)
        except Exception as e:
            logger.warning(f"Experience retrieval failed: {e}")
            return "(No experience history yet)"

        if not experiences:
            return "(No similar past experiences yet)"

        lines = []
        for e in experiences:
            outcome = "worked well" if e["reward"] > 0.2 else "did NOT work well" if e["reward"] < -0.2 else "neutral outcome"
            lines.append(
                f"- Request: \"{e['user_text']}\" → action={e['action']} "
                f"(similarity={e['similarity']:.2f}, reward={e['reward']:.2f}, {outcome})"
            )
        return "\n".join(lines)

    def read_priorities(self) -> str:
        """
        Standing priorities — things the agent holds as important across every
        conversation, not just retrieved for this one. Same table the autonomy
        loop reads, so both paths share one opinion.
        """
        try:
            priorities = self.db.get_active_priorities()
        except Exception as e:
            logger.warning(f"Priority retrieval failed: {e}")
            return "(none)"
        if not priorities:
            return "(none set yet)"
        return "\n".join(f"- {p['statement']} (weight {p['weight']})" for p in priorities)

    def build_messages(self, user_text: str, session_id: str,
                       project_path: str = None, memory_str: str = None,
                       extra_context: str = None, source: str = "cli",
                       credentials_list: str = None, projects_list: str = None) -> list:
        """Build messages array for LLM."""
        context = self.read_context(project_path=project_path)

        if extra_context:
            context += "\n\n" + extra_context

        try:
            from calendar_integration import get_upcoming_events, format_events_for_context
            events = get_upcoming_events(hours=4)
            if events:
                context += "\n\n[CALENDAR]\n" + format_events_for_context(events)
        except Exception:
            pass

        # Retrieval-based selection (the "know when to use it" mechanism):
        # relevance is picked by embedding similarity to THIS request, not dumped wholesale.
        # These are independent (different tables; two do their own embed calls, priorities
        # doesn't) — run them concurrently instead of paying for sequential round-trips.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="context-retrieval") as pool:
            semantic_future = pool.submit(self.read_semantic_context, user_text)
            experiences_future = pool.submit(self.read_similar_experiences, user_text)
            priorities_future = pool.submit(self.read_priorities)
            semantic_context = semantic_future.result()
            similar_experiences = experiences_future.result()
            priorities = priorities_future.result()

        memory = memory_str or "(No memories yet)"
        creds = credentials_list or "(None configured)"
        projects = projects_list or "(None tracked)"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            current_time=datetime.now().strftime("%A, %B %d %Y %H:%M"),
            context=context,
            semantic_context=semantic_context,
            memory=memory,
            credentials_list=creds,
            projects_list=projects,
            priorities=priorities,
            similar_experiences=similar_experiences
        )

        if source == "siri":
            system_prompt += VOICE_MODE_ADDENDUM

        history_limit = 15 if source == "siri" else 10
        history = self.db.get_conversations(session_id=session_id, limit=history_limit)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                    content = parsed.get("speak", content)
                except Exception:
                    pass
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})
        return messages
