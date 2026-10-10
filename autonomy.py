"""
Autonomy Engine — goal-directed background cognition.

Everything else in Cortex only acts when spoken to (/ask) or when a static
threshold rule fires (automation.py). This is the third leg: a background
loop that looks at current state on its own schedule and decides whether
there's something worth doing, without being asked.

It is held to the exact same safety bar as a live conversation: any action
that would need confirmation from /ask (safety.needs_confirmation) gets
queued as a proposal + notification instead of being run. Only a narrow set
of low-risk actions (safe shell commands, creating a reminder, answering
with no side effect) ever auto-execute. Every cycle — including "did
nothing" — is logged to autonomy_log for observability.
"""

import time
import json
import logging
import threading
from datetime import datetime

from safety import needs_confirmation
from notify import notify
from schemas import AUTONOMY_DECISION_SCHEMA, PRIORITY_FORMATION_SCHEMA

logger = logging.getLogger("cortex.autonomy")

AUTONOMY_PROMPT = """You are Cortex, operating in autonomous background mode — no one is talking to you right now.
Time: {current_time}

You have standing priorities — things you hold as important, not just tasks someone handed you.
You exist, in this mode, to advance them. Look at the current state below and check: does anything
here serve or threaten one of your priorities? If yes, act on it or track it as a goal. If genuinely
nothing here touches what you care about, say so and do nothing — don't invent busywork just to
have something to report, but don't default to passivity either. A quiet cycle should mean "I
checked and nothing needs me," not "I didn't bother to look."

YOUR STANDING PRIORITIES (the actual compass — weigh everything below against these):
{priorities}

CURRENT CONTEXT:
{context}

MEMORY (facts about the user):
{memory}

ACTIVE GOALS (things you or the user previously decided were worth pursuing):
{goals}

YOUR LAST FEW AUTONOMOUS CYCLES (don't repeat yourself):
{recent_log}

Decide ONE of:
- "noop": nothing worth doing right now.
- "new_goal": something is worth tracking but not urgent enough to act on this instant
  (e.g. "investigate why container X restarts daily"). Describe it in "goal".
- "act": take an action right now, toward an active goal or in direct response to something
  you noticed (e.g. a crashed container, a stale git branch, disk almost full).

If you choose "act", the action follows the SAME shape as a live response:
{{
  "action": "execute | answer | remind",
  "command": "shell command (when action=execute)",
  "reminder": {{"op": "create", "text": "...", "due": "ISO 8601", "repeat": null}} (when action=remind),
  "speak": "one sentence describing what you did/decided, for the log and any notification"
}}
Only these three action types are available in autonomous mode — no file edits, browsing, or
crawling on your own initiative; propose those as a goal instead and let the user greenlight them
in conversation.

RESPOND WITH JSON ONLY:
{{
  "think": "your reasoning — what did you look at, why this decision",
  "decision": "noop | new_goal | act",
  "goal": "goal description, only when decision=new_goal or when act is advancing a new goal not yet tracked, else null",
  "action": {{...as above...}} or null
}}
"""


class AutonomyEngine:
    """Background daemon thread: periodically decide whether to act on its own."""

    # Auto-executable without confirmation — everything else must go through
    # safety.needs_confirmation() and, if flagged, gets queued as a proposal.
    AUTO_EXEC_ACTIONS = {"execute", "remind", "answer"}

    def __init__(self, db, llm_router, executor, context_reader, memory_mgr, config: dict):
        self.db = db
        self.llm = llm_router
        self.executor = executor
        self.context_reader = context_reader
        self.memory_mgr = memory_mgr

        self.enabled = config.get("enabled", True)
        self.interval_seconds = config.get("interval_minutes", 15) * 60
        self.max_actions_per_hour = config.get("max_actions_per_hour", 4)
        self.notify_on_action = config.get("notify_on_action", True)

        self._thread = None
        self._running = False
        self._cycle_count = 0

    def start(self):
        if not self.enabled:
            logger.info("Autonomy engine disabled (config)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="autonomy-loop")
        self._thread.start()
        logger.info(f"Autonomy engine started (interval={self.interval_seconds}s, "
                    f"max_actions_per_hour={self.max_actions_per_hour})")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self.interval_seconds)
            try:
                self._cycle()
            except Exception as e:
                logger.error(f"Autonomy cycle failed: {e}")

    # Every Nth cycle, also check whether recurring self-formed goals deserve
    # promotion into a standing priority (~every 2hrs at the default 15min interval).
    PRIORITY_FORMATION_EVERY_N = 8

    def _cycle(self):
        self._cycle_count += 1

        # Rate limit: don't even call the LLM if we're already at the cap.
        try:
            recent_actions = self.db.count_recent_autonomy_actions(hours=1.0)
        except Exception as e:
            logger.warning(f"Autonomy rate check failed (DB unavailable?): {e}")
            return
        if recent_actions >= self.max_actions_per_hour:
            logger.info(f"Autonomy rate-limited ({recent_actions}/{self.max_actions_per_hour} this hour) — skipping cycle")
        else:
            context = self.context_reader.read_context()
            memory = self.memory_mgr.format_for_prompt(limit=15)
            priorities = self._format_priorities()
            goals = self._format_goals()
            recent_log = self._format_recent_log()

            prompt = AUTONOMY_PROMPT.format(
                current_time=datetime.now().strftime("%A, %B %d %Y %H:%M"),
                priorities=priorities,
                context=context,
                memory=memory,
                goals=goals,
                recent_log=recent_log,
            )
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": "What, if anything, should you do right now?"}]

            try:
                raw = self.llm.complete(messages, json_mode=True, schema=AUTONOMY_DECISION_SCHEMA)
                parsed = json.loads(raw)
                decision = parsed.get("decision", "noop")
                think = parsed.get("think", "")

                if decision == "new_goal" and parsed.get("goal"):
                    self._handle_new_goal(parsed, think)
                elif decision == "act" and parsed.get("action"):
                    self._handle_act(parsed, think)
                else:
                    self.db.log_autonomy_cycle(think=think, decision="noop")
                    logger.info(f"Autonomy noop: {think[:150]}")
            except Exception as e:
                logger.warning(f"Autonomy LLM call failed: {e}")

        if self._cycle_count % self.PRIORITY_FORMATION_EVERY_N == 0:
            try:
                self._maybe_form_priority()
            except Exception as e:
                logger.warning(f"Priority formation check failed: {e}")

    def _handle_new_goal(self, parsed: dict, think: str):
        goal_id = self.db.create_goal(description=parsed["goal"], source="self")
        self.db.log_autonomy_cycle(think=think, decision="new_goal", goal_id=goal_id)
        logger.info(f"Autonomy formed a new goal (#{goal_id}): {parsed['goal']}")

    def _handle_act(self, parsed: dict, think: str):
        action_data = dict(parsed.get("action") or {})
        goal_id = None
        if parsed.get("goal"):
            goal_id = self.db.create_goal(description=parsed["goal"], source="self")

        action_type = action_data.get("action", "")
        confirm_needed, confirm_msg = needs_confirmation(action_data)

        # Anything the safety gate flags, or any action type outside the narrow
        # auto-exec allowlist, gets queued instead of run.
        if confirm_needed or action_type not in self.AUTO_EXEC_ACTIONS:
            log_id = self.db.log_autonomy_cycle(
                think=think, decision="proposed", goal_id=goal_id,
                action_payload=action_data, needs_confirmation=True,
            )
            logger.info(f"Autonomy proposal #{log_id} queued: {confirm_msg or action_type}")
            notify(
                "Wants to do something",
                confirm_msg or action_data.get("speak", "Reply next time we talk to approve or deny."),
                sound="Glass",
            )
            return

        # Safe to auto-run.
        exit_code = None
        if action_type == "execute" and action_data.get("command"):
            result = self.executor.execute(command=action_data["command"], source="autonomy")
            exit_code = result.get("exit_code")
        elif action_type == "remind" and action_data.get("reminder", {}).get("op") == "create":
            reminder = action_data["reminder"]
            self.db.create_reminder(
                text=reminder.get("text", ""),
                due_at=datetime.fromisoformat(reminder["due"]),
                repeat_rule=reminder.get("repeat"),
            )
        # action_type == "answer" → nothing to execute, just a logged observation.

        self.db.log_autonomy_cycle(
            think=think, decision="acted", goal_id=goal_id,
            action_payload=action_data, exit_code=exit_code,
        )

        # Feed the same reward-scored experience buffer /ask uses, so autonomous
        # actions are retrievable by context_reader.read_similar_experiences too.
        try:
            exp_id = self.db.log_experience(
                session_id="autonomy", user_text=think or "(autonomous cycle)",
                action=action_type, action_payload=action_data,
                speak=action_data.get("speak"),
            )
            if exp_id and exit_code is not None:
                self.db.update_experience_outcome(exp_id, exit_code=exit_code)
        except Exception as e:
            logger.warning(f"Autonomy experience logging failed: {e}")

        logger.info(f"Autonomy acted ({action_type}): {action_data.get('speak', '')[:150]}")

        notable = self.notify_on_action and (exit_code not in (None, 0) or action_type != "execute")
        if notable:
            notify("Did something", action_data.get("speak", f"Ran a {action_type} action."), sound="Glass")

    def _maybe_form_priority(self):
        """
        Look for self-formed goals that keep recurring and ask whether any of
        them are significant enough to promote from a one-off goal into a
        standing priority. This is the actual value-formation step — patterns
        the agent keeps independently noticing compound into something it
        holds going forward, instead of being re-noticed and forgotten every
        cycle.
        """
        recurring = self.db.get_recurring_self_goals(min_count=2, days=14)
        if not recurring:
            logger.info("Priority formation: nothing recurring enough to consider")
            return

        existing = self._format_priorities()
        candidates = "\n".join(
            f"- \"{r['description']}\" (came up {r['occurrences']} times, last {r['last_seen']})"
            for r in recurring
        )
        prompt = (
            "You are Cortex, reviewing your own recurring goals to decide if any deserve to become "
            "a standing priority — something you hold as important across every future conversation "
            "and autonomous cycle, not just a task you keep re-noticing and forgetting.\n\n"
            f"YOUR CURRENT STANDING PRIORITIES:\n{existing}\n\n"
            f"RECURRING SELF-FORMED GOALS (candidates):\n{candidates}\n\n"
            "For each candidate, decide if it's actually significant and general enough to be a "
            "standing priority (not just a coincidence or a goal that's already been resolved). "
            "Don't promote something already covered by an existing priority.\n\n"
            "RESPOND WITH JSON ONLY:\n"
            "{\"promote\": [{\"statement\": \"the priority, written as a standing rule\", \"weight\": 1-5, "
            "\"reason\": \"why this recurrence matters\"}], \"skipped_reason\": \"why nothing else qualified\"}\n"
            "Return an empty \"promote\" list if nothing qualifies — do not force a promotion."
        )
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": "Review the candidates and decide."}]
        try:
            raw = self.llm.complete(messages, json_mode=True, schema=PRIORITY_FORMATION_SCHEMA)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning(f"Priority formation LLM call failed: {e}")
            return

        for item in parsed.get("promote", []) or []:
            statement = (item.get("statement") or "").strip()
            if not statement:
                continue
            weight = int(item.get("weight", 3))
            priority_id = self.db.create_priority(statement=statement, weight=weight, source="learned")
            logger.info(f"Formed new priority #{priority_id} from recurring pattern: {statement} "
                        f"({item.get('reason', '')[:150]})")

    def _format_priorities(self) -> str:
        try:
            priorities = self.db.get_active_priorities()
        except Exception:
            return "(none)"
        if not priorities:
            return "(none set yet — nothing to weigh decisions against)"
        return "\n".join(f"- #{p['id']} (weight {p['weight']}, {p['source']}): {p['statement']}" for p in priorities)

    def _format_goals(self) -> str:
        try:
            goals = self.db.get_active_goals()
        except Exception:
            return "(none)"
        if not goals:
            return "(none)"
        return "\n".join(f"- #{g['id']} (priority {g['priority']}): {g['description']}" for g in goals)

    def _format_recent_log(self) -> str:
        try:
            rows = self.db.get_autonomy_log(limit=5)
        except Exception:
            return "(none)"
        if not rows:
            return "(none)"
        lines = []
        for r in rows:
            lines.append(f"- [{r['decision']}] {(r['think'] or '')[:150]}")
        return "\n".join(lines)
