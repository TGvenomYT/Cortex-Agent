"""
subagents.py — Cortex splitting a job across focused helper agents, only when the TASK
calls for it.

Most requests ("what's my disk usage?", "draft a letter") are best handled by one agent
in one pass — a helper would just burn extra model calls. A request with several
independent parts ("check my disk, search the web for X, and summarise my unread mail")
is different: separate helpers, each with a small focused prompt and its own context,
work through the parts and Cortex stitches the findings together. That isolation is
what keeps a small local model from drowning in a long multi-part conversation.

How the decision is made (cheapest check first, so simple chat pays nothing):
  1. mode:    Settings → Background → Subagents  (auto | off | eager)
  2. signals: a no-LLM complexity score — enumerated parts, several distinct domains
              (web / this Mac / server / email / files / charts / documents), "compare",
              "for each", explicit "in parallel" / "use helpers"…
  3. planner: only if the score clears the bar, ONE model call that either splits the
              request into 2–N independent subtasks or vetoes it. If the model can't
              produce a plan but the user clearly enumerated the parts, we split those.
  4. capacity: low battery / thermal pressure / tight memory → stay single-agent. A local
              model can't truly run helpers in parallel (one set of weights), so they run
              one after another; cloud models fan out (default up to 3 at once).

Helpers get a narrower tool set (no email-send, file edits, reminders, memory writes,
scheduling, or further helpers) and go through the same gate as everything else, as if
nobody were watching: only read-only work runs, and anything that would need the user's
approval is reported back instead of run.
"""
from __future__ import annotations

import re
import json
import time
import queue
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("cortex.subagents")

MODES = ("auto", "off", "eager")

# Tools a helper may OFFER. What actually runs is still decided by safety.evaluate().
HELPER_TOOLS = {"execute_shell", "web_search", "browse_url", "browser_automate", "crawl_site",
                "make_chart", "image_search", "email_check", "email_search", "run_project"}

_DONE = object()

# ── complexity signals ────────────────────────────────────────────────────
_DOMAINS = {
    "web":     r"\b(search the web|look up|google|research|latest|news|find out|online|website|browse|compare .{0,30} (to|with|vs))\b",
    "machine": r"\b(disk|cpu|ram|memory usage|battery|uptime|processes|my mac|this mac|laptop|storage|wi-?fi)\b",
    "server":  r"\b(my server|the server|myserver|ssh|docker|containers?|nginx|server logs?)\b",
    "email":   r"\b(e-?mails?|inbox|gmail|unread)\b",
    "files":   r"\b(files?|folders?|repo|repository|project|codebase|unit tests?|git|readme)\b",
    "chart":   r"\b(plot|graph|chart|visuali[sz]e)\b",
    "docs":    r"\b(pdf|uploaded|documents?|summari[sz]e)\b",
}
_FORCE = re.compile(r"\b(in parallel|at the same time|simultaneously|separately|sub-?agents?|helpers?|"
                    r"spawn (an? )?(agent|helper)s?|delegate)\b", re.I)
_FORBID = re.compile(r"\b(don'?t|do not|no need to|without) (use |spawn |split|delegate|need )?"
                     r"(sub-?agents?|helpers?|delegat|splitting)|\b(yourself|single agent|just you)\b", re.I)
_ENUM_LINE = re.compile(r"^\s*(\d+[.)]|[-*•])\s+\S", re.M)
_ENUM_INLINE = re.compile(r"(\(\s*\d+\s*\)|\b\d\)\s)|\b(first|firstly)\b.+\b(second|secondly|then)\b", re.I | re.S)
_COMPARE = re.compile(r"\bcompare\b|\bfor each\b|\beach of\b|\ball of the following\b|\bboth\b.+\band\b", re.I)
_CONNECT = re.compile(r"\b(and then|after that|also|as well as|plus|along with)\b|;", re.I)
_TRIVIAL = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|yes|no|yep|nope|sure|bye)\b[\s.!?]*$", re.I)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "delegate": {"type": "boolean"},
        "reason": {"type": "string"},
        "subtasks": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "prompt": {"type": "string"}},
            "required": ["title", "prompt"]}},
    },
    "required": ["delegate", "reason", "subtasks"],
}

_PLANNER_PROMPT = (
    "You decide whether a request is worth splitting across helper agents.\n"
    "Split ONLY when it has 2 to {max_n} clearly INDEPENDENT parts — parts that don't need each "
    "other's results — and doing them one after another in a single conversation would be slower "
    "or messier.\n"
    "Do NOT split: a single-topic request, a simple question, chit-chat, or work where each step "
    "depends on the previous one. When in doubt, don't split.\n"
    "For each part write a complete, self-contained instruction a helper can follow with no other "
    "context (name the exact thing to check, find or do).\n"
    'Respond as JSON: {{"delegate": true|false, "reason": "one short sentence", '
    '"subtasks": [{{"title": "3-5 words", "prompt": "..."}}]}}. '
    'If delegate is false, subtasks must be [].'
)

_HELPER_PROMPT = (
    "You are a focused helper agent working for {who}. You were given ONE part of a larger job — "
    "do only that part, using your tools when you need real information. You can't ask the user "
    "anything. When you're done, reply with the findings only: concrete facts, numbers, names, "
    "under 120 words, no greetings, no commentary about being a helper.\n"
    "If something needs the user's approval, or a tool says NOT RUN, say exactly that in your findings.\n"
    "The larger job (for context only — don't do the other parts): {overall}"
)


class SubagentManager:
    def __init__(self, core):
        self.core = core

    # ── settings ──────────────────────────────────────────────────────────
    def settings(self) -> dict:
        try:
            import cortex.storage.settings_store as settings_store
            s = dict(settings_store.section("agents"))
        except Exception:
            s = {}
        mode = s.get("subagents", "auto")
        return {
            "mode": mode if mode in MODES else "auto",
            "max_parallel": int(s.get("max_parallel", 0) or 0),
            "max_helpers": max(2, min(6, int(s.get("max_helpers", 4) or 4))),
            "helper_steps": max(1, min(8, int(s.get("helper_steps", 4) or 4))),
        }

    # ── signals ───────────────────────────────────────────────────────────
    @staticmethod
    def score(text: str) -> tuple[int, list[str]]:
        """A no-LLM measure of "does this request have several independent parts?"."""
        t = (text or "").strip()
        words = len(t.split())
        if words < 6 or _TRIVIAL.match(t):
            return 0, []
        if _FORBID.search(t):
            return -10, ["you asked me to do it myself"]
        pts, why = 0, []
        if _FORCE.search(t):
            pts += 4; why.append("you asked for helpers / parallel work")
        if len(_ENUM_LINE.findall(t)) >= 2 or _ENUM_INLINE.search(t):
            pts += 3; why.append("a list of separate parts")
        if _COMPARE.search(t):
            pts += 2; why.append("a comparison / per-item job")
        c = len(_CONNECT.findall(t))
        if c:
            pts += min(2, c); why.append("several linked asks")
        doms = [d for d, rx in _DOMAINS.items() if re.search(rx, t, re.I)]
        if len(doms) >= 3:
            pts += 3; why.append(f"touches {len(doms)} areas ({', '.join(doms[:4])})")
        elif len(doms) == 2:
            pts += 2; why.append(f"touches 2 areas ({', '.join(doms)})")
        if words > 70:
            pts += 2
        elif words > 35:
            pts += 1
        return pts, why

    @staticmethod
    def split_enumerated(text: str) -> list[dict]:
        """Deterministic fallback: split a clearly enumerated request into its parts."""
        items = [m.strip() for m in re.split(r"^\s*(?:\d+[.)]|[-*•])\s+", text or "", flags=re.M)[1:] if m.strip()]
        if len(items) < 2:
            items = [x.strip(" ,.;") for x in re.split(r"\(\s*\d+\s*\)|\b\d\)\s", text or "") if len(x.strip()) > 8][1:]
        return [{"title": it[:36].rstrip(" ,.;:") or f"Part {i+1}", "prompt": it} for i, it in enumerate(items)] \
            if len(items) >= 2 else []

    # ── capacity ──────────────────────────────────────────────────────────
    def _is_local(self) -> bool:
        try:
            from cortex.llm.llm import OllamaBackend
            return isinstance(getattr(self.core.llm, "primary", self.core.llm), OllamaBackend)
        except Exception:
            return True

    def capacity(self, n: int) -> dict:
        """How many helpers can run at once right now, and whether we should split at all."""
        import cortex.util.power as power
        cfg = self.settings()
        constrained, why = power.is_constrained()
        local = self._is_local()
        if cfg["max_parallel"] > 0:
            par = cfg["max_parallel"]
        else:
            par = 1 if local else min(3, n)
        if constrained:
            par = 1
        return {"parallel": max(1, min(par, n)), "constrained": constrained, "why": why, "local": local}

    # ── the decision ──────────────────────────────────────────────────────
    def assess(self, text: str) -> dict:
        """Should this request be split? Returns
        {delegate, reason, subtasks, parallel, note?} — `note` is a short trace line
        explaining a noteworthy 'no' (resources, switched off)."""
        cfg = self.settings()
        no = lambda reason, note=None: {"delegate": False, "reason": reason, "subtasks": [], "note": note}
        score, why = self.score(text)
        explicit = bool(_FORCE.search(text or "")) and score >= 0

        if cfg["mode"] == "off":
            return no("subagents are off",
                      "Helpers are switched off (Settings → Background → Subagents)." if explicit else None)
        bar = 2 if cfg["mode"] == "eager" else 3
        if score < bar:
            return no(f"simple enough to do directly (score {score})")

        cap = self.capacity(cfg["max_helpers"])
        if cap["constrained"] and cfg["mode"] != "eager":
            return no(f"resources tight ({cap['why']})",
                      f"Staying single-agent: {cap['why']} — splitting would only add load.")

        subtasks, reason = [], ""
        try:
            raw = self.core.llm.complete(
                [{"role": "system", "content": _PLANNER_PROMPT.format(max_n=cfg["max_helpers"])},
                 {"role": "user", "content": text[:2000]}],
                json_mode=True, schema=PLAN_SCHEMA)
            plan = json.loads(raw)
            reason = (plan.get("reason") or "").strip()
            if plan.get("delegate"):
                subtasks = self._clean(plan.get("subtasks"), cfg["max_helpers"])
            elif not explicit:
                return no(reason or "the planner judged it a single job")
        except Exception as e:
            logger.info(f"subagent planner unavailable ({e}) — using signals only")
        if len(subtasks) < 2:
            subtasks = self._clean(self.split_enumerated(text), cfg["max_helpers"])
            reason = reason or "; ".join(why) or "the request lists separate parts"
        if len(subtasks) < 2:
            return no("couldn't find 2+ independent parts to hand out")
        par = self.capacity(len(subtasks))["parallel"]
        return {"delegate": True, "reason": reason or "; ".join(why), "subtasks": subtasks,
                "parallel": par, "note": None}

    def assess_tool_call(self, text: str, subtasks) -> dict:
        """The model itself asked to delegate (the `delegate` tool). Trust it more than a cold
        guess, but still honour the off-switch, the resource gate, and don't let a simple
        request be split."""
        cfg = self.settings()
        if cfg["mode"] == "off":
            return {"delegate": False, "reason": "Helpers are switched off — do this yourself."}
        score, _ = self.score(text)
        if cfg["mode"] == "auto" and score < 2:
            return {"delegate": False, "reason": "That request is simple enough — do it yourself directly."}
        subs = self._clean(subtasks, cfg["max_helpers"])
        if len(subs) < 2:
            return {"delegate": False, "reason": "Need at least 2 independent parts to hand out — do it yourself."}
        cap = self.capacity(len(subs))
        if cap["constrained"] and cfg["mode"] != "eager":
            return {"delegate": False, "reason": f"Resources are tight ({cap['why']}) — do it yourself."}
        return {"delegate": True, "reason": "model requested helpers", "subtasks": subs, "parallel": cap["parallel"]}

    @staticmethod
    def _clean(raw, max_n: int) -> list[dict]:
        out, seen = [], set()
        for i, s in enumerate(raw or []):
            if isinstance(s, str):
                s = {"title": s[:36], "prompt": s}
            prompt = (s.get("prompt") or "").strip()
            if not prompt or prompt.lower() in seen:
                continue
            seen.add(prompt.lower())
            out.append({"title": (s.get("title") or f"Part {i+1}").strip()[:48], "prompt": prompt[:1200]})
        return out[:max_n]

    # ── running helpers ───────────────────────────────────────────────────
    def _helper_tools(self) -> list:
        tools = []
        for t in self.core._tools():
            name = (t.get("function") or {}).get("name", "")
            if name in HELPER_TOOLS or name.startswith("mcp__"):
                tools.append(t)
        return tools

    def _run_helper(self, idx: int, sub: dict, task_text: str, source: str, emit, deadline: float) -> dict:
        from cortex.llm.tools import tool_call_to_action_data
        from cortex.core import safety
        from cortex.core.cortex_core import untrusted
        core, cfg = self.core, self.settings()
        tools = self._helper_tools()
        allowed = {(t.get("function") or {}).get("name") for t in tools}

        sys_prompt = _HELPER_PROMPT.format(who=core._owner, overall=task_text[:500])
        srv = core._server_context()
        if srv:
            sys_prompt += "\n\n" + srv
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": sub["prompt"]}]
        summary, steps, asked_approval, tainted = "", 0, False, False

        for _ in range(cfg["helper_steps"]):
            if time.time() > deadline:
                break
            try:
                result = core.llm.complete_with_tools(messages, tools=tools)
            except Exception as e:
                return {"ok": False, "summary": f"couldn't run this part: {core._llm_error_message(e)}", "steps": steps}
            calls = result.get("tool_calls") or []
            if not calls:
                summary = (result.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": result.get("content", "") or "", "tool_calls": calls})
            for tc in calls:
                fn = tc.get("function") or {}
                name, tcid = fn.get("name", ""), tc.get("id") or ""
                if name not in allowed:
                    obs = f"tool '{name}' isn't available to helpers"
                else:
                    if name.startswith("mcp__"):
                        action = {"action": "mcp", "mcp_name": name, "mcp_args": fn.get("arguments") or {}}
                    else:
                        action = tool_call_to_action_data(tc)
                    verdict = safety.evaluate(action, tainted=tainted, background=True,
                                              mcp_readonly=core._mcp_readonly(action))
                    kind = action.get("action")
                    if verdict.kind == safety.BLOCK:
                        obs = f"REFUSED — {verdict.message}"
                    elif verdict.kind == safety.ASK:
                        asked_approval = True
                        obs = f"NOT RUN — needs the user's approval first: {verdict.message}"
                    elif kind == "chart":
                        src = core._render_chart(action.get("code", ""))
                        if src:
                            emit({"type": "image", "src": src, "caption": action.get("caption", "")})
                        obs = "chart rendered and shown to the user" if src else "chart render failed"
                    elif kind == "image_search":
                        items = core._image_search(action.get("query", ""))
                        if items:
                            emit({"type": "images", "items": items[:8], "query": action.get("query", "")})
                        obs = f"showed {len(items)} web images" if items else "no images found"
                    else:
                        emit({"type": "subagent", "id": idx, "state": "step",
                              "name": kind or "?", "detail": core._tool_detail(action)})
                        obs, outside = core.observe(action, source)
                        tainted = tainted or outside
                        obs = untrusted(kind, str(obs)[:2300]) if outside else obs
                steps += 1
                messages.append({"role": "tool", "tool_call_id": tcid, "content": str(obs)[:2500]})
        if not summary:   # ran out of steps/time mid-work — ask for what it has so far
            try:
                messages.append({"role": "user", "content": "Out of time. Report what you found so far, "
                                                            "under 120 words, findings only."})
                summary = (core.llm.complete(messages) or "").strip()
            except Exception:
                summary = ""
        if not summary:
            return {"ok": False, "summary": "I didn't get a result for this part.", "steps": steps,
                    "tainted": tainted}
        return {"ok": True, "summary": summary, "steps": steps, "needs_approval": asked_approval,
                "tainted": tainted}

    def run_stream(self, subtasks: list[dict], task_text: str, source: str, parallel: int):
        """Generator: yields live `subagent`/`image` events while helpers work, then a final
        {"type": "_results", "results": [...]} (internal; the caller consumes it)."""
        n = len(subtasks)
        q: queue.Queue = queue.Queue()
        results: list = [None] * n
        deadline = time.time() + 300
        for i, s in enumerate(subtasks):
            q.put({"type": "subagent", "id": i, "title": s["title"], "state": "queued"})

        def work(i: int):
            q.put({"type": "subagent", "id": i, "title": subtasks[i]["title"], "state": "start"})
            try:
                res = self._run_helper(i, subtasks[i], task_text, source, q.put, deadline)
            except Exception as e:
                logger.exception("helper crashed")
                res = {"ok": False, "summary": f"this part hit an error: {e}", "steps": 0}
            results[i] = res
            q.put({"type": "subagent", "id": i, "state": "done", "ok": res["ok"],
                   "summary": res["summary"][:700], "steps": res.get("steps", 0)})
            q.put(_DONE)

        pool = ThreadPoolExecutor(max_workers=max(1, parallel), thread_name_prefix="helper")
        try:
            for i in range(n):
                pool.submit(work, i)
            finished = 0
            while finished < n:
                ev = q.get()
                if ev is _DONE:
                    finished += 1
                else:
                    yield ev
        finally:
            pool.shutdown(wait=False)
        yield {"type": "_results", "results": results}

    @staticmethod
    def format_findings(subtasks: list[dict], results: list, task_text: str) -> str:
        lines = ["[HELPER RESULTS] You split the user's request across helper agents. Their findings:"]
        for s, r in zip(subtasks, results):
            r = r or {"summary": "(no result)"}
            lines.append(f"\n### {s['title']}\n{r['summary']}")
        lines.append("\nNow answer the user's ORIGINAL request using these findings — one clear, "
                     "combined reply. Don't talk about helpers unless a part failed or needs the "
                     "user's approval (then say what and ask).")
        return "\n".join(lines)
