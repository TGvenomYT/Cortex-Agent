"""
autonomy_session.py — multi-step, tool-using autonomous cognition.

The original AutonomyEngine (autonomy.py) did ONE thing per wake: read state,
make a single decision (noop / new_goal / act), run at most one low-risk
action. This replaces that with a genuine agentic session — a
plan -> act -> observe -> reflect loop that can chain several steps toward a
goal in a single wake, use real tools (the local agentkit catalog), manage its
own goals, and score its own outcome afterwards.

Design constraints (unchanged from the single-shot version, tightened where
noted):
  * Same safety bar as a live /ask. Destructive or outward-facing actions are
    NEVER auto-run; they are queued as proposals + a quiet notification.
  * "Expanded auto-exec with guardrails": a wider set of read-only / reversible
    actions now auto-run (read-only agentkit queries, safe shell, reminders,
    internal goal bookkeeping, quiet notifications). Everything else is queued.
  * DB-backed rate limiting across invocations (works in the wake-run-exit
    model, where in-memory counters would reset every run).
  * Every step — including a quiet "nothing to do" — is logged to autonomy_log.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime

from cortex.core.safety import needs_confirmation
from cortex.util.notify import notify

logger = logging.getLogger("cortex.autonomy.session")

# ── agentkit (owned tool gateway) — optional, imported lazily/gracefully ──
_AGENTKIT_PATH = os.path.expanduser("~/agent-tools")

# Read-only / reversible agentkit tools that may auto-run unsupervised.
# Anything NOT in this set (blackwall offensive recon, guardian ban controls,
# cortex_remember writes, …) is proposed and queued instead.
READONLY_AGENTKIT = {
    "devops_status", "devops_containers", "devops_health", "devops_security",
    "devops_exposure", "devops_logins", "devops_logs", "devops_ask",
    "cortex_memory_search", "shell_explain", "shell_suggest", "shell_recall",
    "breach_scan_me",
}


def _load_agentkit():
    """Return (registry, available_tool_names) or (None, set())."""
    if _AGENTKIT_PATH not in sys.path:
        sys.path.insert(0, _AGENTKIT_PATH)
    try:
        from agentkit import registry  # type: ignore
        return registry, set(registry.names())
    except Exception as e:
        logger.info(f"agentkit unavailable ({e}); autonomy runs without external tools")
        return None, set()


PLANNER_PROMPT = """You are Cortex in autonomous background mode — no one is talking to you right now.
Time: {current_time}

You act on your own standing priorities. Each wake, you may take SEVERAL steps:
look at state, call a tool, read the result, then decide the next step — or stop.
Do real work toward your priorities and goals; never invent busywork. A quiet
wake ("I looked and nothing needs me") is a perfectly good outcome.

YOUR STANDING PRIORITIES (your compass):
{priorities}

ACTIVE GOALS:
{goals}

CURRENT CONTEXT:
{context}

MEMORY (facts about the user):
{memory}

YOUR LAST FEW AUTONOMOUS CYCLES (don't repeat yourself):
{recent_log}

OBSERVATIONS THIS SESSION SO FAR (results of steps you already took this wake):
{observations}

AVAILABLE TOOLS you may call this step (name — what it does):
{tools}

Choose your NEXT step. Respond with JSON ONLY:
{{
  "think": "what you're looking at and why this step",
  "step": "act | finish",
  "action": {{
     "kind": "tool | shell | remind | goal | notify | answer",
     "tool": "<agentkit tool name>   (kind=tool)",
     "args": {{...}}                  (kind=tool),
     "command": "<shell command>"     (kind=shell),
     "reminder": {{"text":"...","due":"ISO-8601","repeat":null}}  (kind=remind),
     "goal": {{"op":"create|complete|drop","id":<int|null>,"description":"...","result":"..."}} (kind=goal),
     "message": "one quiet sentence to surface to the user"  (kind=notify),
     "speak": "one sentence describing this step, for the log"
  }} (null when step=finish),
  "done_reason": "why you're stopping (only when step=finish)"
}}

Rules: prefer read-only tools to understand before changing anything. Destructive
or outward-facing actions will be held for the user's approval, not run — so if
something needs that, do a "goal" create to track it and move on. Stop
("finish") as soon as there's nothing more worth doing this wake."""


REFLECT_PROMPT = """You are Cortex, reflecting on the autonomous session you just ran.

WHAT YOU DID THIS WAKE (in order):
{transcript}

Score the session honestly and decide if anything should change going forward.
Respond with JSON ONLY:
{{
  "summary": "one or two sentences: what you accomplished or concluded",
  "reward": -1.0 to 1.0 (did this advance a priority/goal? negative if it was wasted or wrong),
  "followups": ["goal description", ...]   (new goals worth tracking; [] if none)
}}"""


class AutonomySession:
    AUTO_EXEC_KINDS = {"tool", "shell", "remind", "goal", "notify", "answer"}

    def __init__(self, db, llm, executor, context_reader, memory_mgr, config: dict):
        self.db = db
        self.llm = llm
        self.executor = executor
        self.context_reader = context_reader
        self.memory_mgr = memory_mgr
        self.max_actions_per_hour = config.get("max_actions_per_hour", 4)
        self.max_steps = config.get("max_steps_per_wake", 4)
        self.notify_on_action = config.get("notify_on_action", True)
        self.registry, self.tool_names = _load_agentkit()

    # ── public entry ──────────────────────────────────────────────────────
    def run(self) -> dict:
        """Run one autonomous session (possibly several steps). Returns a summary dict."""
        try:
            recent = self.db.count_recent_autonomy_actions(hours=1.0)
        except Exception as e:
            logger.warning(f"rate check failed (DB down?): {e}")
            return {"status": "skipped", "reason": "db_unavailable"}
        if recent >= self.max_actions_per_hour:
            logger.info(f"rate-limited ({recent}/{self.max_actions_per_hour}/hr) — skipping")
            return {"status": "rate_limited"}

        observations: list[str] = []
        transcript: list[dict] = []
        actions_taken = 0
        self._seen: set[str] = set()  # action signatures already run this wake

        for step_no in range(1, self.max_steps + 1):
            budget_left = self.max_actions_per_hour - (recent + actions_taken)
            if budget_left <= 0:
                break
            parsed = self._decide(observations)
            if parsed is None:
                break
            think = parsed.get("think", "")
            if parsed.get("step") == "finish" or not parsed.get("action"):
                self.db.log_autonomy_cycle(think=think or parsed.get("done_reason", ""),
                                           decision="noop")
                logger.info(f"session finished at step {step_no}: {parsed.get('done_reason', think)[:150]}")
                break

            outcome = self._run_step(parsed["action"], think)
            transcript.append({"think": think, **outcome})
            observations.append(outcome["observation"])
            if outcome["executed"]:
                actions_taken += 1
            if outcome.get("stop"):
                break

        summary = self._reflect(transcript) if transcript else {"summary": "quiet wake", "reward": 0.0}
        logger.info(f"session done: {actions_taken} action(s). {summary.get('summary','')[:150]}")
        return {"status": "ok", "actions": actions_taken, "summary": summary}

    # ── one planning step ────────────────────────────────────────────────
    def _decide(self, observations: list[str]) -> dict | None:
        prompt = PLANNER_PROMPT.format(
            current_time=datetime.now().strftime("%A, %B %d %Y %H:%M"),
            priorities=self._fmt_priorities(),
            goals=self._fmt_goals(),
            context=self._safe(self.context_reader.read_context),
            memory=self._safe(lambda: self.memory_mgr.format_for_prompt(limit=15)),
            recent_log=self._fmt_recent_log(),
            observations="\n".join(f"- {o}" for o in observations) or "(none yet)",
            tools=self._fmt_tools(),
        )
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": "What is your next step?"}]
        try:
            raw = self.llm.complete(messages, json_mode=True)
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"planner LLM call failed: {e}")
            return None

    # ── execute / queue one action ───────────────────────────────────────
    def _run_step(self, action: dict, think: str) -> dict:
        kind = (action.get("kind") or "").strip()
        speak = action.get("speak", "")

        # Dedup: a small model tends to repeat the same read over and over.
        # Refuse an identical action twice in one wake and tell it to move on.
        sig = self._signature(kind, action)
        if sig in self._seen:
            return {"executed": False, "kind": kind,
                    "observation": f"SKIPPED (already did '{sig}' this wake — "
                                   "do something different or finish)."}
        self._seen.add(sig)

        allowed, why = self._auto_exec_allowed(kind, action)
        if not allowed:
            self.db.log_autonomy_cycle(think=think, decision="proposed",
                                       action_payload=action, needs_confirmation=True)
            notify("Wants to do something", why or speak or "Approve next time we talk.",
                   sound="Glass")
            return {"executed": False, "observation": f"PROPOSED (queued, needs approval): {why}",
                    "kind": kind}

        try:
            observation, exit_code = self._execute(kind, action)
        except Exception as e:
            observation, exit_code = f"error: {e}", 1

        self.db.log_autonomy_cycle(think=think, decision="acted",
                                   action_payload=action, exit_code=exit_code)
        try:
            exp_id = self.db.log_experience(session_id="autonomy",
                                            user_text=think or "(autonomous step)",
                                            action=kind, action_payload=action,
                                            speak=speak)
            if exp_id and exit_code is not None:
                self.db.update_experience_outcome(exp_id, exit_code=exit_code)
        except Exception as e:
            logger.debug(f"experience logging skipped: {e}")

        if self.notify_on_action and kind == "notify":
            pass  # already notified inside _execute
        return {"executed": True, "observation": observation, "kind": kind, "exit_code": exit_code}

    def _auto_exec_allowed(self, kind: str, action: dict) -> tuple[bool, str]:
        if kind not in self.AUTO_EXEC_KINDS:
            return False, f"unknown action kind '{kind}'"
        if kind == "tool":
            name = action.get("tool", "")
            if name not in self.tool_names:
                return False, f"unknown tool '{name}'"
            if name not in READONLY_AGENTKIT:
                return False, f"tool '{name}' is not read-only — needs approval"
            return True, ""
        if kind == "shell":
            cmd = action.get("command", "") or ""
            need, msg = needs_confirmation({"action": "execute", "command": cmd})
            return (not need), (msg or "")
        if kind == "remind":
            return True, ""
        if kind == "goal":
            return True, ""
        if kind == "notify":
            return True, ""
        if kind == "answer":
            return True, ""
        return False, "unclassified action"

    def _execute(self, kind: str, action: dict) -> tuple[str, int | None]:
        if kind == "tool":
            name = action.get("tool", "")
            args = action.get("args") or {}
            result = self.registry.call(name, args)
            text = json.dumps(result, default=str)
            return (f"{name} -> {text[:500]}", 0)

        if kind == "shell":
            cmd = action["command"]
            res = self.executor.execute(command=cmd, source="autonomy")
            return (f"ran `{cmd}` (exit {res.get('exit_code')}): "
                    f"{str(res.get('output',''))[:400]}", res.get("exit_code"))

        if kind == "remind":
            r = action.get("reminder") or {}
            self.db.create_reminder(text=r.get("text", ""),
                                    due_at=datetime.fromisoformat(r["due"]),
                                    repeat_rule=r.get("repeat"))
            return (f"created reminder: {r.get('text','')}", 0)

        if kind == "goal":
            g = action.get("goal") or {}
            op = g.get("op")
            if op == "create":
                gid = self.db.create_goal(description=g.get("description", ""), source="self")
                return (f"tracked new goal #{gid}: {g.get('description','')}", 0)
            if op == "complete":
                self.db.update_goal(g["id"], status="completed", result=g.get("result"))
                return (f"completed goal #{g.get('id')}", 0)
            if op == "drop":
                self.db.update_goal(g["id"], status="dropped", result=g.get("result", "no longer relevant"))
                return (f"dropped goal #{g.get('id')}", 0)
            return (f"unknown goal op '{op}'", 1)

        if kind == "notify":
            msg = action.get("message", action.get("speak", ""))
            notify("Cortex noticed", msg, sound=None)  # quiet: no sound, no speech
            try:
                self.db.add_nudge("Cortex noticed", msg[:600], kind="info", source="autonomy")
            except Exception:
                pass
            return (f"notified: {msg}", 0)

        if kind == "answer":
            return (f"observed: {action.get('speak','')}", 0)

        return (f"no-op for kind '{kind}'", None)

    # ── reflection / self-scoring ────────────────────────────────────────
    def _reflect(self, transcript: list[dict]) -> dict:
        lines = [f"- [{t.get('kind')}] {t.get('observation','')[:200]}" for t in transcript]
        prompt = REFLECT_PROMPT.format(transcript="\n".join(lines))
        try:
            raw = self.llm.complete([{"role": "system", "content": prompt},
                                     {"role": "user", "content": "Reflect."}],
                                    json_mode=True)
            parsed = json.loads(raw)
        except Exception as e:
            logger.debug(f"reflection skipped: {e}")
            return {"summary": "(reflection failed)", "reward": 0.0}

        for desc in (parsed.get("followups") or []):
            if isinstance(desc, str) and desc.strip():
                try:
                    self.db.create_goal(description=desc.strip(), source="self")
                except Exception:
                    pass
        return parsed

    # ── priority formation (promote recurring goals) ─────────────────────
    def maybe_form_priority(self):
        """Promote recurring self-formed goals into standing priorities."""
        try:
            recurring = self.db.get_recurring_self_goals(min_count=2, days=14)
        except Exception:
            return
        if not recurring:
            return
        existing = self._fmt_priorities()
        candidates = "\n".join(
            f'- "{r["description"]}" (x{r["occurrences"]}, last {r["last_seen"]})' for r in recurring)
        prompt = (
            "You are Cortex reviewing recurring self-formed goals to decide if any deserve to "
            "become a STANDING PRIORITY (held across every future wake), not a task you keep "
            "re-noticing.\n\n"
            f"CURRENT PRIORITIES:\n{existing}\n\nRECURRING CANDIDATES:\n{candidates}\n\n"
            'RESPOND JSON ONLY: {"promote":[{"statement":"...","weight":1-5,"reason":"..."}]}. '
            "Empty list if nothing qualifies; don't duplicate an existing priority.")
        try:
            parsed = json.loads(self.llm.complete(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "Decide."}], json_mode=True))
        except Exception:
            return
        for item in parsed.get("promote", []) or []:
            stmt = (item.get("statement") or "").strip()
            if stmt:
                pid = self.db.create_priority(statement=stmt,
                                              weight=int(item.get("weight", 3)), source="learned")
                logger.info(f"formed priority #{pid}: {stmt}")

    # ── formatting helpers ───────────────────────────────────────────────
    def _fmt_tools(self) -> str:
        if not self.registry:
            return "(no external tools available; use shell/remind/goal/notify/answer)"
        lines = []
        for t in self.registry.TOOLS:
            tag = "" if t.name in READONLY_AGENTKIT else "  [needs approval]"
            lines.append(f"- {t.name}: {t.description}{tag}")
        return "\n".join(lines)

    def _fmt_priorities(self) -> str:
        try:
            ps = self.db.get_active_priorities()
        except Exception:
            return "(none)"
        if not ps:
            return "(none set yet)"
        return "\n".join(f"- #{p['id']} (w{p['weight']}, {p['source']}): {p['statement']}" for p in ps)

    def _fmt_goals(self) -> str:
        try:
            gs = self.db.get_active_goals()
        except Exception:
            return "(none)"
        return "\n".join(f"- #{g['id']} (p{g['priority']}): {g['description']}" for g in gs) or "(none)"

    def _fmt_recent_log(self) -> str:
        try:
            rows = self.db.get_autonomy_log(limit=5)
        except Exception:
            return "(none)"
        return "\n".join(f"- [{r['decision']}] {(r['think'] or '')[:120]}" for r in rows) or "(none)"

    @staticmethod
    def _signature(kind: str, action: dict) -> str:
        if kind == "tool":
            return f"tool:{action.get('tool')}:{json.dumps(action.get('args') or {}, sort_keys=True)}"
        if kind == "shell":
            return f"shell:{(action.get('command') or '').strip()}"
        if kind == "goal":
            g = action.get("goal") or {}
            return f"goal:{g.get('op')}:{g.get('id')}:{(g.get('description') or '')[:60]}"
        if kind == "remind":
            return f"remind:{(action.get('reminder') or {}).get('text','')[:60]}"
        if kind == "notify":
            return f"notify:{(action.get('message') or action.get('speak') or '')[:60]}"
        return f"{kind}:{(action.get('speak') or '')[:60]}"

    @staticmethod
    def _safe(fn) -> str:
        try:
            return fn()
        except Exception as e:
            return f"(unavailable: {e})"
