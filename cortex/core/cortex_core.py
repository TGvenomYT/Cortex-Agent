"""
cortex_core.py — the in-process agent brain of Cortex. No server required.

This is what lets Cortex *complete a job*, not just answer once. Given a task it
runs a multi-step loop:

    plan (LLM picks a tool)  ->  act (run it)  ->  observe (feed result back)  ->  repeat

until the model answers without calling a tool (= done) or a step budget is hit.
It uses the real tool surface — shell, a real browser (Playwright), web fetch/
search, file edits, project runs, reminders — and the warm "brother" persona
from context_reader. Runs fully offline (local SQLite + local Ollama), so it
works with Tailscale off.

Safety: every action goes through safety.evaluate(). What can't just run is held. With a
person present the job pauses on approval buttons and carries on from the same point once
approved; with nobody there (scheduled tasks, agentkit) the action goes to the approvals
inbox and the job continues without it. Anything read from outside is wrapped as
untrusted data and marks the turn "tainted", which tightens the gate for the rest of it.
"""
from __future__ import annotations

import os
import re
import json
import time
import logging
from datetime import datetime

from cortex.core import safety

logger = logging.getLogger("cortex.core")

# Someone is at the keyboard to answer an approval prompt.
HUMAN_SOURCES = {"gui", "cli", "voice", "siri"}
# Tools whose results contain text written by someone else.
OUTSIDE_TOOLS = {"search", "browse", "browser", "crawl", "email_check", "email_search", "mcp"}
_FETCH_CMD = re.compile(r"\b(curl|wget|gh)\b")
_TURN_BUDGET = 600      # seconds one job may spend in the agent loop

_FINAL_SYNTH = (
    "You just worked on a task for {who}. Here's a log of what you did and found.\n"
    "Give them the result in a couple of warm, plain sentences — no jargon, no raw "
    "output, lead with the outcome. If the task is done, say so; if something blocked "
    "you, say what and suggest the next move.\n\nWORK LOG:\n{log}\n\n"
    'Respond as JSON: {{"speak": "..."}}'
)


def untrusted(source: str, text: str) -> str:
    """Mark text as data from outside. The system prompt tells the model never to take
    instructions from inside these tags; a closing tag in the text can't break out."""
    clean = (text or "").replace("</untrusted_content", "<\\/untrusted_content")
    return f'<untrusted_content source="{source}">\n{clean}\n</untrusted_content>'


class _TurnRecorder:
    """Watches the event stream of one turn so the reply can be stored WITH what made it
    rich — the thinking/tool/helper trace, charts, image grids — and a reloaded chat looks
    like it did live (it used to come back as bare text)."""
    _TRACE = ("thinking", "tool", "observation", "subagent", "note")
    _MAX_EVENTS = 60

    def __init__(self):
        self.trace, self.media, self.answer = [], [], None

    def feed(self, ev: dict) -> None:
        t = ev.get("type")
        if t in self._TRACE:
            if len(self.trace) < self._MAX_EVENTS:
                self.trace.append({k: v for k, v in ev.items() if k != "type"} | {"t": t})
        elif t == "image":
            self.media.append({"t": "image", "src": ev.get("src"), "caption": ev.get("caption", "")})
        elif t == "images":
            self.media.append({"t": "images", "items": ev.get("items", [])[:8], "query": ev.get("query", "")})
        elif t == "answer":
            self.answer = ev.get("text") or ""

    def parts(self) -> list:
        out = []
        if self.trace:
            out.append({"t": "trace", "events": self.trace})
        return out + self.media


class _Job:
    """What one job carries through the agent loop. Survives an approval pause (to_dict)."""
    def __init__(self, text: str, session_id: str, source: str, tainted: bool = False,
                 work_log: list = None, step: int = 0, chart_shown: bool = False):
        self.text, self.session_id, self.source = text, session_id, source
        self.tainted = tainted
        self.work_log = work_log if work_log is not None else []
        self.step = step
        self.chart_shown = chart_shown
        self.notices: list[str] = []
        self.llm_failed = False
        self.exit_code = None

    @property
    def background(self) -> bool:
        return self.source not in HUMAN_SOURCES

    def to_dict(self) -> dict:
        return {"text": self.text, "tainted": self.tainted, "work_log": self.work_log[-20:],
                "step": self.step, "chart_shown": self.chart_shown}


class CortexCore:
    def __init__(self, config: dict = None):
        from cortex.util.cortex_bootstrap import load_config, load_secrets, build_for_autonomy
        load_secrets()
        self.config = config or load_config()
        deps = build_for_autonomy(self.config)
        if deps is None:
            raise RuntimeError("Cortex could not open its data store.")
        self.db = deps["db"]
        self.llm = deps["llm"]
        self.executor = deps["executor"]
        self.context_reader = deps["context_reader"]
        self.memory_mgr = deps["memory_mgr"]
        self._owner = self.config.get("owner", {}).get("name", "you")
        agent_cfg = self.config.get("agent", {})
        self.max_steps = agent_cfg.get("max_steps", 12)
        self.background_max_steps = agent_cfg.get("background_max_steps", 6)
        self._grants: dict[str, set] = {}     # session_id -> what "allow in this chat" covers
        self._last_experience: dict[str, int] = {}
        from cortex.core.subagents import SubagentManager
        self.subagents = SubagentManager(self)
        # MCP: connect to any configured MCP servers (lazy — loaded on first use)
        self.mcp = None
        try:
            import cortex.storage.settings_store as settings_store
            from cortex.integrations.mcp_client import MCPManager
            self.mcp = MCPManager(settings_store.load().get("mcp_servers", []))
        except Exception as e:
            logger.info(f"MCP unavailable: {e}")

    def _server_context(self) -> str:
        """Inject the user's configured server so the agent uses it (not a hardcoded IP)."""
        try:
            import cortex.storage.settings_store as settings_store
            s = settings_store.load().get("server", {})
            if s.get("host"):
                hosts = (self.config.get("remote_hosts") or {}).values()
                key = next((h.get("key") for h in hosts if h.get("ip") == s["host"] and h.get("key")), None)
                opt = f"-i {key} " if key else ""
                return (f"[CONFIGURED SERVER] '{s.get('label','server')}' — host {s['host']}, "
                        f"ssh user {s.get('user','')}. For 'my server'/'the server', use: "
                        f"ssh {opt}{s.get('user','')}@{s['host']} \"<cmd>\".")
        except Exception:
            pass
        return None

    def _render_chart(self, code: str, approved: bool = False) -> str:
        import uuid
        import cortex.llm.viz as viz
        if not approved and not viz.check_code(code)[0]:
            logger.warning("chart code refused: it does more than plot")
            return None
        art = os.path.expanduser("~/.cortex/artifacts")
        os.makedirs(art, exist_ok=True)
        fn = uuid.uuid4().hex[:12] + ".png"
        ok, err = viz.render_matplotlib(code, os.path.join(art, fn))
        if ok:
            return "/artifact/" + fn
        logger.warning(f"chart render failed: {err}")
        return None

    def _llm_error_message(self, e: Exception) -> str:
        """Turn an LLM failure into something the user can act on, naming the real cause
        (bad key, model not found, rate limit…) instead of a generic 'can't reach a model'."""
        import httpx
        model = getattr(getattr(self.llm, "primary", None), "model", "the selected model")
        if isinstance(e, httpx.HTTPStatusError):
            code = e.response.status_code
            try:
                detail = (e.response.json().get("error") or {})
                detail = detail.get("message") if isinstance(detail, dict) else str(detail)
            except Exception:
                detail = (e.response.text or "")[:200]
            detail = (detail or "").strip()[:240]
            if code in (401, 403):
                why = "the API key was rejected — re-check it in Settings → AI Providers"
            elif code == 404:
                why = f"`{model}` isn't available for your key — pick a different model from the selector"
            elif code == 429:
                why = "rate limit or quota reached — wait a bit or check your plan/billing"
            elif code == 400:
                why = f"the provider rejected this request for `{model}` — try another model"
            else:
                why = f"the provider returned an error ({code})"
            return f"The model provider said no: {why}." + (f"\n\n> {detail}" if detail else "")
        if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
            return ("I can't reach a model. Either start Ollama on this computer, or add an "
                    "**Ollama Cloud** key (or any other provider) under **Settings → AI Providers** "
                    "and pick that model from the selector top-left.")
        return f"I hit an error talking to the model: {str(e)[:200]}"

    _CHART_ASK = re.compile(r"\b(plot|graph|chart|draw|sketch|visuali[sz]e|histogram|diagram)\b", re.I)
    _NOT_CHART = re.compile(r"\b(the|this|that|movie|film|book|story|main|whole)\s+plot\b(?!\s+(for|showing))"
                            r"|\bplot\s+(twist|summary|hole|line)s?\b|\bdraw\s+(on|from|up|attention|conclusions?)\b",
                            re.I)
    _CLAIMS_CHART = re.compile(r"\b(here'?s|here is|below is|see) (the|a|your) (graph|plot|chart|diagram)\b"
                               r"|\b(graph|plot|chart) (below|above)\b|\bi('ve| have) (plotted|drawn|graphed|charted)\b",
                               re.I)

    def _wants_chart(self, text: str) -> bool:
        t = text or ""
        return bool(self._CHART_ASK.search(t)) and not self._NOT_CHART.search(t)

    def _chart_code_for(self, text: str) -> str:
        """Ask the model for ONLY matplotlib code — the reliable fallback for when
        the model narrated a plot instead of calling make_chart (common on models
        whose tool-calling is weak, e.g. cloud variants)."""
        prompt = (
            "Write ONLY Python matplotlib code to create this plot: " + text + "\n"
            "`plt` (matplotlib.pyplot) and `np` (numpy) are ALREADY imported and available. "
            "Do NOT import anything. Do NOT call plt.show() or plt.savefig(). "
            "Add a title, axis labels, a grid, and a legend if there are multiple series. "
            "For functions with asymptotes (e.g. hyperbolas, 1/x, tan), mask the discontinuity "
            "so the curve doesn't draw a vertical spike. Output ONLY the code — no prose, no ``` fences.")
        try:
            raw = (self.llm.complete([{"role": "user", "content": prompt}]) or "").strip()
            if "```" in raw:
                m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.S)
                if m:
                    raw = m.group(1).strip()
            return raw or None
        except Exception as e:
            logger.warning(f"chart code generation failed: {e}")
            return None

    def _image_search(self, query: str) -> list:
        try:
            from ddgs import DDGS
            res = DDGS().images(query, max_results=8)
            return [{"url": r.get("image"), "thumb": r.get("thumbnail") or r.get("image"),
                     "title": r.get("title", ""), "page": r.get("url", "")} for r in res]
        except Exception as e:
            logger.info(f"image_search: {e}")
            return []

    def _rag_context(self, query: str) -> str:
        """Pull the most relevant chunks from the user's uploaded files (RAG).

        Only inject chunks that clear a relevance threshold — otherwise an
        unrelated query (e.g. 'what's my disk usage') pulls in whatever docs
        happen to rank highest and feeds the model irrelevant noise."""
        try:
            import cortex.storage.rag as rag
            hits = rag.retrieve(self.db, query, k=6)
            hits = [h for h in hits if (h.get("similarity") or 0) >= 0.4]
            if hits:
                return ("[FROM THE USER'S UPLOADED FILES — use/cite when relevant]\n" + untrusted(
                    "uploaded_files", "\n---\n".join(f"({h['source']}) {h['chunk']}" for h in hits)))
        except Exception as e:
            logger.info(f"rag retrieve: {e}")
        return None

    def _tools(self):
        from cortex.llm.tools import TOOLS
        extra = []
        if self.mcp:
            try:
                extra = self.mcp.openai_tools()
            except Exception as e:
                logger.info(f"MCP tools load failed: {e}")
        tools = TOOLS
        if self.subagents.settings()["mode"] == "off":      # don't even offer it
            tools = [t for t in tools if (t.get("function") or {}).get("name") != "delegate"]
        return tools + extra

    def _obs_limit(self) -> int:
        """How much of a tool's output to show the model. A cloud model has room for far
        more than a small local one, which starts dropping its instructions when flooded."""
        local = getattr(self.llm, "is_local", lambda: True)()
        return 3000 if local else 10000

    # ── public: non-streaming (consumes the stream) ────────────────────────
    def ask(self, text: str, session_id: str = "cli", source: str = "cli") -> dict:
        final = {"action": "answer", "speak": "All done.", "steps": 0}
        for ev in self.ask_stream(text, session_id=session_id, source=source):
            if ev.get("type") == "final":
                final = ev["action"]
        return final

    def resolve(self, session_id: str, pending_id: int, decision: str, source: str = "cli") -> dict:
        """Answer an approval prompt without streaming (the terminal uses this)."""
        final = {"action": "answer", "speak": "Nothing was waiting for an answer.", "steps": 0}
        for ev in self.resolve_stream(session_id, pending_id, decision, source):
            if ev.get("type") == "final":
                final = ev["action"]
        return final

    # ── persistence ───────────────────────────────────────────────────────
    def _save_turn(self, session_id: str, role: str, content: str, parts: list = None) -> None:
        """Write one chat row. Loud on failure (it used to be `except: pass`, so a lost turn
        left no trace) and retried once — a transient 'database is locked' shouldn't eat a message."""
        for attempt in (1, 2):
            try:
                self.db.save_conversation(session_id, role, content, parts=parts)
                return
            except Exception as e:
                logger.warning(f"couldn't save {role} turn (attempt {attempt}): {e}")
                time.sleep(0.4)

    def _persisted(self, session_id: str, user_text: str, events):
        """Same event stream as `events`, plus crash-proof persistence: the user's message
        is stored the instant it arrives (a refresh or crash mid-reply can't lose it), and
        the reply is stored — with its trace/charts — however the turn ends."""
        self._save_turn(session_id, "user", user_text)
        rec = _TurnRecorder()
        failure = None
        try:
            for ev in events:
                rec.feed(ev)
                yield ev
        except GeneratorExit:
            raise                     # consumer walked away; nothing more to store
        except Exception as e:
            failure = e
            logger.exception("ask_stream failed")
            raise
        finally:
            if rec.answer is not None:
                self._save_turn(session_id, "assistant", rec.answer, parts=rec.parts())
            elif failure is not None:
                self._save_turn(session_id, "assistant", f"Something went wrong: {str(failure)[:200]}")

    # ── public: streaming — yields live events so a UI/CLI can show thinking ─
    def ask_stream(self, text: str, session_id: str = "cli", source: str = "cli"):
        """Generator of events as Cortex works:
          {"type":"thinking","text":...}      model reasoning before a step
          {"type":"delta","text":...}         a piece of the reply as the model writes it
          {"type":"tool","name":...,"detail":...}  a tool is about to run
          {"type":"observation","text":...}   what the tool returned (short)
          {"type":"answer","text":...}        the final plain-language reply
          {"type":"approval","id":...,"text":...,"can_always":bool}  waiting for the user
          {"type":"final","action":{...}}     full action dict (for ask())
        """
        return self._persisted(session_id, text, self._ask_stream(text, session_id, source))

    _DECISION_LABEL = {"approve": "Yes, do it", "approve_session": "Yes, and allow this for the rest of the chat",
                       "reject": "No, cancel"}

    def resolve_stream(self, session_id: str, pending_id: int, decision: str, source: str = "gui"):
        """The user pressed an approval button. Matched by id, so it can only ever answer the
        prompt it was shown on, and no typed text is involved."""
        label = self._DECISION_LABEL.get(decision)
        if not label:
            raise ValueError(f"unknown decision {decision!r}")

        def events():
            pending = self.db.get_pending_action(session_id)
            if not pending or pending.get("id") != pending_id:
                msg = "That request has already been handled or has expired — ask me again if you still want it."
                yield {"type": "answer", "text": msg}
                yield {"type": "final", "action": {"action": "answer", "speak": msg, "steps": 0}}
                return
            yield from self._resume(pending, decision, session_id, source)
        return self._persisted(session_id, label, events())

    def _ask_stream(self, text: str, session_id: str = "cli", source: str = "cli"):
        # A previous turn left something awaiting this session's yes/no. Typed answers are
        # only honoured from a person: a scheduled prompt must never approve its own run.
        if source in HUMAN_SOURCES:
            pending = None
            try:
                pending = self.db.get_pending_action(session_id)
            except Exception as e:
                logger.debug(f"pending-action lookup failed: {e}")
            if pending:
                verdict = self._classify_confirmation(text)
                if verdict in ("yes", "no"):
                    decision = "approve" if verdict == "yes" else "reject"
                    yield from self._resume(pending, decision, session_id, source)
                    return
                # anything else: the user moved on — drop the stale draft (it was never
                # sent/run) and handle `text` as a normal new message below.
                self.db.clear_pending_action(session_id)

        yield {"type": "status", "text": "thinking…"}
        try:
            memory_str = self.memory_mgr.format_for_prompt(limit=15)
        except Exception:
            memory_str = None
        rag = self._rag_context(text)
        extra = "\n\n".join(x for x in [self._server_context(), rag] if x) or None
        messages = self.context_reader.build_messages(
            user_text=text, session_id=session_id, memory_str=memory_str, source=source,
            extra_context=extra,
        )
        job = _Job(text, session_id, source, tainted=bool(rag))

        # Should this job be split across helper agents? Decided per task — a cheap
        # signal check first, a planner call only if the task looks multi-part.
        try:
            plan = self.subagents.assess(text)
        except Exception as e:
            logger.warning(f"subagent assessment failed: {e}")
            plan = None
        if plan and plan.get("note"):
            yield {"type": "note", "text": plan["note"]}
        if plan and plan.get("delegate"):
            subs = plan["subtasks"]
            yield {"type": "status", "text": f"splitting this across {len(subs)} helpers…"}
            pace = " (one after another)" if plan["parallel"] == 1 else f" ({plan['parallel']} at a time)"
            yield {"type": "note", "text": f"Splitting into {len(subs)} helpers{pace} — {plan['reason']}"}
            results = None
            for ev in self.subagents.run_stream(subs, text, source, plan["parallel"]):
                if ev.get("type") == "_results":
                    results = ev["results"]
                else:
                    yield ev
            messages.append({"role": "user",
                             "content": self.subagents.format_findings(subs, results or [], text)})
            self._absorb_helpers(job, subs, results)

        yield from self._agent_loop(messages, job)

    @staticmethod
    def _absorb_helpers(job: _Job, subs: list, results: list) -> None:
        for st, r in zip(subs, results or []):
            job.work_log.append(f"[helper: {st['title']}] {(r or {}).get('summary', '')}")
            job.tainted = job.tainted or bool((r or {}).get("tainted"))

    def _model_step(self, messages: list, tools: list):
        """One model call as ('delta', text)* then ('done', result)."""
        stream = getattr(self.llm, "stream_with_tools", None)
        if stream is None:
            yield "done", self.llm.complete_with_tools(messages, tools=tools)
        else:
            yield from stream(messages, tools)

    def _agent_loop(self, messages: list, job: _Job):
        from cortex.llm.tools import tool_call_to_action_data, rescue_tool_calls
        tools_list = self._tools()
        known = {(t.get("function") or {}).get("name") for t in tools_list}
        limit = self.background_max_steps if job.background else self.max_steps
        deadline = time.time() + _TURN_BUDGET
        mode = safety.current_mode()
        grants = self._grants.get(job.session_id, set())
        last_action, final_speak, held = {"action": "answer"}, None, None
        ask_options, asked = [], False
        retried_empty = False

        while job.step < limit and time.time() < deadline and not held and not asked:
            job.step += 1
            result = None
            try:
                for kind, payload in self._model_step(messages, tools_list):
                    if kind == "delta":
                        yield {"type": "delta", "text": payload}
                    else:
                        result = payload
            except Exception as e:
                logger.error(f"LLM unavailable: {e}")
                final_speak = self._llm_error_message(e)
                job.llm_failed = True
                break

            result = rescue_tool_calls(result or {}, known)
            think = (result.get("thinking") or result.get("content") or "").strip()
            tool_calls = result.get("tool_calls") or []

            if not tool_calls:
                final_speak = (result.get("content") or "").strip()
                if not final_speak and not retried_empty:     # some models stop on an empty turn
                    retried_empty = True
                    messages.append({"role": "user", "content": "Please continue — finish the job or answer me."})
                    continue
                final_speak = final_speak or None
                break

            if think:
                yield {"type": "thinking", "text": think[:600]}

            messages.append({"role": "assistant", "content": result.get("content", "") or "",
                             "tool_calls": tool_calls})

            for i, tc in enumerate(tool_calls):
                _tcid = tc.get("id") or ""   # OpenAI/Anthropic need tool results to reference this
                _nm = (tc.get("function") or {}).get("name", "")
                if _nm.startswith("mcp__"):
                    action = {"action": "mcp", "mcp_name": _nm,
                              "mcp_args": (tc.get("function") or {}).get("arguments") or {}}
                else:
                    action = tool_call_to_action_data(tc)
                last_action = action

                def reply(content):
                    messages.append({"role": "tool", "tool_call_id": _tcid, "tool_name": _nm, "content": content})

                if action.get("action") == "ask":   # the model needs the user to decide
                    if job.background:
                        reply("Nobody is here to answer. Decide yourself or say what you need in your reply.")
                        continue
                    final_speak = action.get("question") or "Quick question —"
                    ask_options = action.get("options") or []
                    asked = True
                    reply("asked the user; their answer comes as the next message")
                    for later in tool_calls[i + 1:]:
                        messages.append({"role": "tool", "tool_call_id": later.get("id") or "",
                                         "tool_name": (later.get("function") or {}).get("name", ""),
                                         "content": "skipped — waiting for the user"})
                    break

                verdict = safety.evaluate(action, tainted=job.tainted, background=job.background, grants=grants,
                                          mode=mode, mcp_readonly=self._mcp_readonly(action))
                if verdict.kind == safety.BLOCK:
                    logger.warning(f"refused: {action.get('action')} {verdict.message}")
                    yield {"type": "note", "text": verdict.message}
                    job.work_log.append(f"[refused] {verdict.message}")
                    reply(f"REFUSED — {verdict.message} Don't try to work around this; tell the user.")
                    continue
                if verdict.kind == safety.ASK:
                    if job.background:
                        pid = self._queue_for_approval(action, verdict.message, job)
                        job.work_log.append(f"[held for approval #{pid}] {action.get('action')}")
                        yield {"type": "note", "text": "Waiting for your OK in the approvals inbox."}
                        reply("NOT RUN — this needs the user's approval and they aren't here. It's in their "
                              "approvals inbox. Carry on without it and mention that it's waiting.")
                        continue
                    for later in tool_calls[i + 1:]:
                        messages.append({"role": "tool", "tool_call_id": later.get("id") or "",
                                         "tool_name": (later.get("function") or {}).get("name", ""),
                                         "content": "skipped — an earlier step is waiting for the user's "
                                                    "approval. Do this again if it's still needed."})
                    held = (action, verdict, _tcid, _nm)
                    final_speak = verdict.message
                    break

                obs = yield from self._perform(action, job)
                reply(obs)

        if held:
            action, verdict, tcid, name = held
            pid = self.db.save_pending_action(
                job.session_id, action.get("action", ""),
                {**action, "_resume": {"messages": messages, "tool_call_id": tcid, "tool_name": name,
                                       "job": job.to_dict(), "grant": verdict.grant}},
                final_speak)
            approval = {"id": pid, "text": final_speak, "can_always": bool(verdict.grant),
                        "scope": verdict.grant}
            last_action = {**action, "speak": final_speak, "action": action.get("action", "answer"),
                           "steps": len(job.work_log), "approval": approval}
            self._finish(job.session_id, job.text, last_action)
            yield {"type": "answer", "text": final_speak}
            yield {"type": "approval", **approval}
            yield {"type": "final", "action": last_action}
            return

        yield from self._finish_turn(job, last_action, final_speak, asked, ask_options)

    def _finish_turn(self, job: _Job, last_action: dict, final_speak, asked: bool, ask_options: list):
        # Guaranteed-chart fallback: the user asked for a plot but the model never
        # produced one (it narrated "here's the graph" without calling make_chart —
        # common on weaker/cloud tool-callers). Generate the code directly and render
        # it, and NEVER let the reply claim a graph that doesn't exist. A good answer
        # that makes no such claim is left alone if the chart can't be made.
        if not asked and not job.background and not job.llm_failed and not job.chart_shown \
                and self._wants_chart(job.text):
            yield {"type": "status", "text": "rendering the chart…"}
            code = self._chart_code_for(job.text)
            src = self._render_chart(code) if code else None
            if src:
                job.chart_shown = True
                yield {"type": "image", "src": src, "caption": ""}
                final_speak = final_speak or "Here's the graph."
                last_action = {"action": "chart"}
            elif not final_speak or self._CLAIMS_CHART.search(final_speak):
                final_speak = ("I couldn't render that plot — the code my model produced didn't "
                               "run. Tell me the exact equation or range and I'll try again.")

        if final_speak is None:
            yield {"type": "status", "text": "wrapping up…"}
            final_speak = self._final_summary(job.work_log)
        if job.notices:
            final_speak += "\n\n" + "\n\n".join(job.notices)

        last_action["speak"] = final_speak
        last_action.setdefault("action", "answer")
        last_action["steps"] = len(job.work_log)
        if job.llm_failed:
            last_action["error"] = True
        self._finish(job.session_id, job.text, last_action, exit_code=job.exit_code, learn=not job.background)
        yield {"type": "answer", "text": final_speak}
        if asked:
            yield {"type": "ask", "options": ask_options}
        elif job.source == "gui" and not job.llm_failed and self._want_followups(final_speak):
            fu = self._followups(job.text, final_speak)
            if fu:
                yield {"type": "followups", "items": fu}
        yield {"type": "final", "action": last_action}

    def _mcp_readonly(self, action: dict) -> bool:
        if action.get("action") != "mcp" or not self.mcp:
            return False
        try:
            return self.mcp.is_readonly(action.get("mcp_name", ""))
        except Exception:
            return False

    def _perform(self, action: dict, job: _Job, approved: bool = False):
        """Run one allowed action, yielding its live events; returns what the model sees."""
        kind = action.get("action")
        if kind == "chart":  # render a plot inline
            src = self._render_chart(action.get("code", ""), approved=approved)
            if src:
                job.chart_shown = True
                yield {"type": "image", "src": src, "caption": action.get("caption", "")}
            obs = "chart rendered and shown to the user" if src else "chart render failed"
            job.work_log.append(f"[chart] {obs}")
            return obs
        if kind == "image_search":  # web images inline
            items = self._image_search(action.get("query", ""))
            if items:
                yield {"type": "images", "items": items[:8], "query": action.get("query", "")}
            obs = (f"showed {len(items)} web images for '{action.get('query')}'"
                   if items else "no images found")
            job.work_log.append(f"[image_search] {obs}")
            return obs
        if kind == "delegate":   # the model asked for helpers itself
            verdict = self.subagents.assess_tool_call(job.text, action.get("subtasks"))
            if not verdict["delegate"]:
                yield {"type": "note", "text": f"Not splitting: {verdict['reason']}"}
                return verdict["reason"]
            subs, results = verdict["subtasks"], None
            yield {"type": "note", "text": f"Splitting into {len(subs)} helpers"}
            for ev in self.subagents.run_stream(subs, job.text, job.source, verdict["parallel"]):
                if ev.get("type") == "_results":
                    results = ev["results"]
                else:
                    yield ev
            self._absorb_helpers(job, subs, results)
            return self.subagents.format_findings(subs, results or [], job.text)[:6000]

        yield {"type": "tool", "name": kind or "?", "detail": self._tool_detail(action)}
        obs, outside = self.observe(action, job.source)
        job.tainted = job.tainted or outside
        job.work_log.append(f"[{kind}] {obs}")
        if kind in ("execute", "devops", "run_project"):
            m = re.search(r"-> exit (-?\d+)", obs)
            job.exit_code = int(m.group(1)) if m else job.exit_code
        if kind == "remind" and (action.get("reminder") or {}).get("op", "create") == "create" \
                and not self._reminders_running():
            job.notices.append("Heads up: reminders are switched off in Settings → Background, so this "
                               "one won't fire until you turn them on.")
        yield {"type": "observation", "text": obs[:400]}
        wire = obs[:self._obs_limit()]
        return untrusted(kind, wire) if outside else wire

    def observe(self, action: dict, source: str) -> tuple[str, bool]:
        """Run a tool. -> (result, whether that result contains outside content)."""
        kind = action.get("action")
        obs = self._execute_tool(action, source)
        outside = kind in OUTSIDE_TOOLS or (kind in ("execute", "devops")
                                            and bool(_FETCH_CMD.search(action.get("command") or "")))
        return obs, outside

    @staticmethod
    def _reminders_running() -> bool:
        try:
            import cortex.core.workers as workers
            return workers.is_on("reminders")
        except Exception:
            return True

    # ── approvals ─────────────────────────────────────────────────────────
    def _resume(self, pending: dict, decision: str, session_id: str, source: str):
        """The user answered a held action. Either way the job carries on from where it paused."""
        payload = dict(pending["payload"])
        resume = payload.pop("_resume", None) or {}
        self.db.clear_pending_action(session_id)
        approved = decision != "reject"
        exp = self._last_experience.get(session_id)
        if exp:
            try:
                self.db.update_experience_outcome(exp, was_confirmed=approved)
            except Exception as e:
                logger.debug(f"outcome not recorded: {e}")

        saved = resume.get("job") or {}
        job = _Job(saved.get("text", ""), session_id, source, tainted=saved.get("tainted", False),
                   work_log=saved.get("work_log"), step=saved.get("step", 0),
                   chart_shown=saved.get("chart_shown", False))
        messages = resume.get("messages")

        if approved and decision == "approve_session" and resume.get("grant"):
            self._grants.setdefault(session_id, set()).add(resume["grant"])

        if not approved:
            obs = "cancelled by the user before it ran"
            job.work_log.append(f"[{payload.get('action')}] {obs}")
            if not messages:
                final = "Okay — cancelled, didn't do that."
                yield {"type": "answer", "text": final}
                yield {"type": "final", "action": {**payload, "speak": final, "steps": 1}}
                return
        else:
            # Approval never overrides a hard refusal, however the prompt got here.
            verdict = safety.evaluate(payload, background=False, mode="balanced", grants=set())
            if verdict.kind == safety.BLOCK:
                yield {"type": "answer", "text": verdict.message}
                yield {"type": "final", "action": {**payload, "speak": verdict.message, "steps": 0}}
                return
            yield {"type": "status", "text": "on it…"}
            obs = yield from self._perform(payload, job, approved=True)

        if not messages:    # an old draft with nothing to resume
            yield {"type": "answer", "text": obs}
            yield {"type": "final", "action": {**payload, "speak": obs, "steps": 1}}
            return
        messages.append({"role": "tool", "tool_call_id": resume.get("tool_call_id") or "",
                         "tool_name": resume.get("tool_name") or "", "content": obs})
        yield from self._agent_loop(messages, job)

    def _queue_for_approval(self, action: dict, message: str, job: _Job) -> int:
        """Nobody's here: put the action in the approvals inbox. An identical one already
        waiting isn't queued twice, so a repeating job can't flood the inbox."""
        existing = self.db.find_open_proposal(action)
        if existing:
            return existing
        pid = self.db.log_autonomy_cycle(think=f"[{job.source}] {job.text[:200]}", decision="proposed",
                                         action_payload=action, needs_confirmation=True)
        try:
            self.db.add_nudge("Needs your OK", (message or "")[:600], kind="warn", source="approval")
        except Exception as e:
            logger.debug(f"approval nudge skipped: {e}")
        return pid

    def list_approvals(self) -> list[dict]:
        """What's waiting in the inbox (scheduled tasks + autonomy)."""
        out = []
        for p in self.db.get_open_proposals():
            action = self._normalize_proposal(p.get("action_payload"))
            if action is None:
                continue
            v = safety.evaluate(action)
            out.append({"id": p["id"], "think": (p.get("think") or "")[:200], "when": str(p.get("cycle_at")),
                        "text": v.message or f"Run {action.get('action')}?", "kind": action.get("action")})
        return out

    @staticmethod
    def _normalize_proposal(payload) -> dict | None:
        """Autonomy stores {kind: shell|tool, …}; tasks store a normal action."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return None
        if not isinstance(payload, dict):
            return None
        kind = payload.get("kind")
        if kind == "shell":
            return {"action": "execute", "command": payload.get("command", "")}
        if kind == "tool":
            return {"action": "agentkit", "tool": payload.get("tool"), "args": payload.get("args") or {}}
        return payload if payload.get("action") else None

    def resolve_approval(self, proposal_id: int, approve: bool) -> dict:
        """Approve or reject one inbox item. -> {ok, message}"""
        p = self.db.get_pending_proposal(proposal_id)
        if not p:
            return {"ok": False, "message": "That one was already handled."}
        action = self._normalize_proposal(p.get("action_payload"))
        if not approve or action is None:
            self.db.resolve_proposal(proposal_id, False)
            return {"ok": True, "message": "Okay, dropped it."}
        if safety.evaluate(action).kind == safety.BLOCK:
            self.db.resolve_proposal(proposal_id, False)
            return {"ok": False, "message": safety.evaluate(action).message}
        if action.get("action") == "agentkit":
            from cortex.core.autonomy_session import _load_agentkit
            registry, names = _load_agentkit()
            if not registry or action.get("tool") not in names:
                self.db.resolve_proposal(proposal_id, False)
                return {"ok": False, "message": f"I can't reach the tool {action.get('tool')}."}
            obs = json.dumps(registry.call(action["tool"], action.get("args") or {}), default=str)[:1500]
        else:
            obs = self._execute_tool(action, "approval")
        self.db.resolve_proposal(proposal_id, True, exit_code=0)
        return {"ok": True, "message": obs}

    # Whole-message answers only. "ok what does that do?", "okay don't send it" and "send it
    # to bob instead" are NOT approvals, and this is deliberately not a prefix match: what
    # runs must not hinge on the first word of a sentence. The buttons don't go through here.
    _YES_REPLIES = {"yes", "y", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "k", "go", "go ahead", "proceed",
                    "confirm", "confirmed", "approve", "approved", "do it", "run", "run it", "send", "send it",
                    "yes do it", "yes please", "yes go ahead", "yes run it", "yes send it", "ok do it",
                    "okay do it", "ok go ahead", "okay go ahead", "ok send it", "ok run it", "please do",
                    "please send", "please send it", "just do it", "just run it", "just send it", "sounds good",
                    "go for it", "yes proceed"}
    _NO_REPLIES = {"no", "n", "nope", "nah", "cancel", "stop", "dont", "do not", "wait", "no cancel",
                   "no thanks", "no dont", "never mind", "nevermind", "hold off", "hold on", "not yet",
                   "abort", "dont do it", "dont send it", "dont run it"}

    @classmethod
    def _classify_confirmation(cls, text: str) -> str:
        """
        yes | no | unclear — a deterministic check, not an LLM call. Whether a drafted
        email actually sends must not depend on the model's mood; it depends on what the
        user typed, and only an unambiguous whole-message answer counts.
        """
        t = re.sub(r"[^a-z0-9 ]+", "", (text or "").lower().replace(",", " "))
        t = " ".join(t.split())
        if t in cls._NO_REPLIES:
            return "no"
        if t in cls._YES_REPLIES:
            return "yes"
        return "unclear"

    @staticmethod
    def _want_followups(answer: str) -> bool:
        """Follow-up chips cost a whole extra model call after EVERY reply. Skip it when the
        user turned it off, the reply is a one-liner, or the machine is short on headroom."""
        try:
            import cortex.storage.settings_store as settings_store
            import cortex.util.power as power
            mode = settings_store.section("resources").get("followups", "auto")
            if mode == "off":
                return False
            if mode == "always":
                return True
            if len(answer or "") < 60:
                return False
            return not power.is_constrained()[0]
        except Exception:
            return True

    def _followups(self, user_text: str, answer: str) -> list:
        """A few short follow-up prompt suggestions (Tab-to-fill in the UI)."""
        try:
            raw = self.llm.complete(
                [{"role": "system", "content":
                  "Suggest 3 short, natural follow-up messages the user might send next, in the "
                  "user's first-person voice, max 7 words each. Return JSON only: {\"items\":[\"...\"]}."},
                 {"role": "user", "content": f"I asked: {user_text}\nYou answered: {answer[:500]}"}],
                json_mode=True)
            items = json.loads(raw).get("items") or []
            return [str(x).strip() for x in items if str(x).strip()][:3]
        except Exception:
            return []

    @staticmethod
    def _tool_detail(action: dict) -> str:
        kind = action.get("action")
        if kind in ("execute", "devops"):
            return action.get("command", "")[:120]
        if kind == "browser":
            return action.get("browser_url", "")
        if kind == "browse":
            return action.get("browse_url", "")
        if kind == "search":
            return action.get("search_query", "")
        if kind == "crawl":
            return action.get("crawl_url", "")
        if kind == "edit":
            return action.get("edit_file", "")
        if kind == "remind":
            return (action.get("reminder") or {}).get("op", "")
        if kind == "email_search":
            return action.get("email_query", "")
        if kind == "email_send":
            return f"to {action.get('email_to','')}"
        if kind == "email_check":
            return "unread across accounts"
        if kind == "remember":
            return action.get("remember_content", "")[:80]
        if kind == "forget":
            return action.get("forget_search") or f"#{action.get('forget_id')}"
        if kind == "mcp":
            return action.get("mcp_name", "").replace("mcp__", "").replace("__", " · ")
        if kind == "workers":
            return " ".join(x for x in (action.get("workers_op"), action.get("worker") or action.get("profile")) if x)
        if kind == "chart":
            return action.get("caption", "") or "plot"
        if kind == "image_search":
            return action.get("query", "")
        if kind == "run_project":
            return f"{action.get('run_project_command', '')} in {action.get('run_project_path', '')}"
        return ""

    # ── unified tool executor: every action -> one observation string ──────
    def _execute_tool(self, action: dict, source: str) -> str:
        kind = action.get("action", "answer")
        try:
            if kind in ("execute", "devops"):
                return self._run_shell(action.get("command", ""), source)
            if kind == "search":
                from cortex.web.websearch import search_and_format
                return str(search_and_format(action.get("search_query", "")))[:2500]
            if kind == "browse":
                return self._run_browse(action)
            if kind == "browser":
                return self._run_browser(action)
            if kind == "crawl":
                from cortex.web.webcrawler import crawl_and_format, crawl_with_credentials
                url = action.get("crawl_url", "")
                cred = action.get("crawl_credential")
                mx = action.get("crawl_max_pages", 3)
                if cred:
                    creds = self.config.get("web_credentials", {}).get(cred, {})
                    return str(crawl_with_credentials(url, creds, max_pages=mx))[:2500]
                return str(crawl_and_format(url, max_pages=mx))[:2500]
            if kind == "edit":
                return self._run_edit(action)
            if kind == "run_project":
                import cortex.core.projects as projects
                return projects.run(action.get("run_project_path", "."),
                                    action.get("run_project_command", ""), self.executor)
            if kind == "mcp":
                return self.mcp.call(action.get("mcp_name", ""), action.get("mcp_args") or {}) if self.mcp else "MCP not available"
            if kind == "email_check":
                return self._email_check()
            if kind == "email_search":
                return self._email_search(action.get("email_query", ""))
            if kind == "email_send":
                return self._email_send(action)
            if kind == "remember":
                return self._remember(action)
            if kind == "forget":
                return self._forget(action)
            if kind == "remind":
                return self._handle_reminder(action.get("reminder") or {})
            if kind == "schedule_task":
                return self._handle_schedule_task(action)
            if kind == "workers":
                return self._handle_workers(action)
            if kind == "credential_prompt":
                return ("Needs a login I don't have stored. Ask the user for the "
                        f"{action.get('credential_prompt_site','site')} credentials next time you talk.")
            return "Nothing to do for that step."
        except Exception as e:
            return f"that step hit an error: {e}"

    def _run_shell(self, command: str, source: str) -> str:
        if not command:
            return "no command given"
        bad, why = safety.is_catastrophic(command)
        if bad:
            logger.warning(f"refused catastrophic command: {command}")
            return f"REFUSED — {why}"
        res = self.executor.execute(command=command, source="cortex-core", timeout=180)
        code = res.get("exit_code")
        room = self._obs_limit()
        out = (res.get("output") or res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        parts = []
        if out:
            parts.append(out[-room:])
        if err:
            parts.append("stderr: " + err[-(room // 3):])
        return f"ran `{command}` -> exit {code}; " + ("\n".join(parts) or "(no output)")

    def _run_browse(self, action: dict) -> str:
        from cortex.web.webbrowser import open_in_browser, fetch_and_format as browse_and_format
        url = action.get("browse_url", "")
        if action.get("browse_open"):
            open_in_browser(url)
            return f"opened {url} in the browser"
        content = str(browse_and_format(url) or "")
        # Auto-fallback to a real browser when static fetch comes back thin (JS site).
        if len(content.strip()) < 200:
            import cortex.web.browser as pw
            if pw.available():
                r = pw.render_page(url)
                if r.get("success"):
                    return f"(rendered) {r.get('title','')}\n{r.get('text','')[:2200]}"
            return content or "page returned almost no static content (likely JavaScript-heavy)."
        return content[:2500]

    def _run_browser(self, action: dict) -> str:
        import cortex.web.browser as pw
        if not pw.available():
            return ("the real browser (Playwright/Chromium) isn't installed yet — "
                    "run: .venv/bin/playwright install chromium")
        url = action.get("browser_url", "")
        steps = action.get("browser_steps") or []
        r = pw.automate(url, steps) if steps else pw.render_page(url)
        if not r.get("success"):
            return f"browser failed: {r.get('error')}"
        parts = [f"{r.get('title','')} @ {r.get('url','')}", (r.get("text", "") or "")[:2000]]
        if r.get("extracted"):
            parts.append("extracted: " + json.dumps(r["extracted"])[:800])
        return "\n".join(parts)

    def _run_edit(self, action: dict) -> str:
        from cortex.util.file_editor import read_file as editor_read_file, apply_edit
        from cortex.llm.schemas import EDIT_SCHEMA
        path = action.get("edit_file", "")
        instruction = action.get("edit_instruction", "")
        fd = editor_read_file(path)
        if not fd.get("success"):
            return f"can't read {path}: {fd.get('error')}"
        try:
            raw = self.llm.complete(
                [{"role": "system", "content":
                  "Given a file and an instruction, produce the edit. Respond as JSON: "
                  '{"find": "exact text to replace", "replace": "new text", "explanation": "what changed"}'},
                 {"role": "user", "content": f"FILE: {path}\nINSTRUCTION: {instruction}\n\nCONTENT:\n{fd['content'][:15000]}"}],
                json_mode=True, schema=EDIT_SCHEMA)
            ed = json.loads(raw)
            res = apply_edit(path, ed["find"], ed["replace"])
            return (f"edited {path}: {ed.get('explanation','')}" if res.get("success")
                    else f"edit failed: {res.get('error')}")
        except Exception as e:
            return f"edit failed: {e}"

    # ── email (Gmail over IMAP/SMTP) ──────────────────────────────────────
    def _email_check(self) -> str:
        import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store
        accts = mail_store.list_accounts()
        if not accts:
            return "no Gmail accounts connected yet — add them in Settings → Email"
        lines = []
        for a in accts[:6]:
            pw = mail_store.password(a["email"])
            if not pw:
                continue
            msgs = mail.fetch_unread(a["email"], pw, limit=6)
            if msgs and "error" in msgs[0]:
                lines.append(f"{a['email']}: couldn't reach ({msgs[0]['error'][:60]})")
            elif msgs:
                subs = "; ".join(f"{m['from']}: {m['subject']}" for m in msgs)
                lines.append(f"{a['email']} — {len(msgs)} unread: {subs}")
            else:
                lines.append(f"{a['email']}: inbox clear")
        return "\n".join(lines) or "no readable accounts"

    def _email_search(self, query: str) -> str:
        import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store
        hits = []
        for a in mail_store.list_accounts()[:6]:
            pw = mail_store.password(a["email"])
            if not pw:
                continue
            for m in mail.search(a["email"], pw, query, limit=8):
                if "error" not in m:
                    hits.append(f"[{a['email']}] {m['from']}: {m['subject']} ({m['date']})")
        return ("found:\n" + "\n".join(hits[:15])) if hits else f"nothing matching '{query}'"

    def _email_send(self, action: dict) -> str:
        import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store
        accts = mail_store.list_accounts()
        frm = action.get("email_from") or (accts[0]["email"] if accts else None)
        if not frm:
            return "no Gmail account to send from"
        pw = mail_store.password(frm)
        if not pw:
            return f"no stored app password for {frm}"
        ok, err = mail.send(frm, pw, action.get("email_to", ""),
                            action.get("email_subject", ""), action.get("email_body", ""))
        return f"sent to {action.get('email_to')} from {frm}" if ok else f"send failed: {err}"

    # ── memory (explicit remember/forget — see cortex/core/memory.py) ──────
    def _remember(self, action: dict) -> str:
        content = (action.get("remember_content") or "").strip()
        if not content:
            return "nothing to remember — empty content"
        mtype = action.get("remember_type") or "fact"
        self.memory_mgr.store_fact(content, mem_type=mtype, source="conversation",
                                   importance=5 if mtype == "rule" else 4)
        return f"remembered ({mtype}): {content}"

    def _forget(self, action: dict) -> str:
        return self.memory_mgr.forget(memory_id=action.get("forget_id"),
                                      search=action.get("forget_search"))

    def _handle_reminder(self, r: dict) -> str:
        op = r.get("op", "create")
        if op == "create":
            try:
                self.db.create_reminder(text=r.get("text", ""),
                                        due_at=datetime.fromisoformat(r["due"]),
                                        repeat_rule=r.get("repeat"))
                return f"set a reminder: {r.get('text','')}"
            except Exception as e:
                return f"couldn't set that reminder ({e})"
        if op == "list":
            try:
                rem = self.db.get_all_reminders(include_fired=False)
            except Exception:
                rem = []
            if not rem:
                return "no reminders set"
            return "reminders: " + "; ".join(f"{x['text']} ({x['due_at']})" for x in rem[:10])
        if op == "delete" and r.get("id"):
            return "deleted reminder" if self.db.delete_reminder(r["id"]) else "no such reminder"
        return "done"

    def _handle_workers(self, a: dict) -> str:
        """Chat control over Cortex's own background workers + the model's memory."""
        import cortex.core.workers as workers
        import cortex.core.resources as resources
        op = a.get("workers_op") or "list"
        if op == "list":
            return workers.describe() + "\n" + resources.describe()
        if op == "unload_model":
            return resources.unload_models()
        if op == "profile":
            prof = a.get("profile")
            r = workers.apply_profile(prof or "")
            if not r.get("ok") and not r.get("profile"):
                return f"couldn't switch profile: {r.get('error') or r.get('errors')}"
            return (f"background profile is now '{r['profile']}'"
                    + (f" (some issues: {'; '.join(r['errors'])})" if r.get("errors") else ""))
        key = workers.resolve_key(a.get("worker") or "")
        if not key:
            return f"I couldn't tell which worker you mean by '{a.get('worker')}' — ask me to list them."
        if op == "run":
            r = workers.run_now(key)
            return f"started {key} now" if r.get("ok") else f"couldn't start {key}: {r.get('error')}"
        if op in ("enable", "disable"):
            r = workers.set_enabled(key, op == "enable")
            return (f"{key} is now {'ON' if r.get('on') else 'off'}" if r.get("ok")
                    else f"couldn't change {key}: {r.get('error') or 'launchd refused'}")
        return "Nothing to do for that step."

    def _handle_schedule_task(self, action: dict) -> str:
        import cortex.core.tasks as tasks
        op = action.get("task_op", "add")
        if op == "list":
            ts = tasks.list_tasks()
            if not ts:
                return "no recurring tasks scheduled"
            return "scheduled tasks: " + "; ".join(
                f"[{t['id']}] {t['desc']} — {t.get('human', t['schedule'])} → {t.get('deliver','notify')}"
                f"{'' if t.get('enabled', True) else ' (paused)'}" for t in ts)
        if op == "remove":
            tid = action.get("task_id")
            if not tid:
                return "need the task id to remove (op=list to see them)"
            tasks.remove_task(tid)
            return f"removed task {tid}"
        # add
        prompt = action.get("task_prompt")
        schedule = action.get("task_schedule")
        if not prompt or not schedule:
            return "need both a prompt and a schedule to add a task"
        try:
            t = tasks.add_task(desc=action.get("task_desc") or "",
                               prompt=prompt, schedule=schedule,
                               deliver=action.get("task_deliver") or "notify")
            return (f"scheduled '{t['desc']}' to run {t['human']}, "
                    f"delivering via {t['deliver']} (id {t['id']})")
        except Exception as e:
            return f"couldn't schedule that task ({e})"

    # ── final plain-language summary when the job ran multi-step ────────────
    def _final_summary(self, work_log: list[str]) -> str:
        if not work_log:
            return "All done."
        prompt = _FINAL_SYNTH.format(who=self._owner, log="\n".join(work_log)[:5000])
        try:
            raw = self.llm.complete(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "Wrap it up for them."}], json_mode=True)
            return json.loads(raw).get("speak") or "Done — handled it."
        except Exception:
            return "Done — handled it."

    def _finish(self, session_id: str, user_text: str, action: dict, exit_code: int = None,
                learn: bool = False) -> dict:
        """Log the experience (learning signal) and give memory its chance to learn. The chat
        rows themselves are written by ask_stream — see _save_turn."""
        try:
            exp = self.db.log_experience(session_id=session_id, user_text=user_text,
                                         action=action.get("action", "answer"),
                                         action_payload={k: v for k, v in action.items()
                                                         if k not in ("speak", "approval")},
                                         speak=action.get("speak"))
            if exp:
                self._last_experience[session_id] = exp
                if exit_code is not None:
                    self.db.update_experience_outcome(exp, exit_code=exit_code)
            self.db.compute_pending_rewards()
        except Exception as e:
            logger.debug(f"experience not logged: {e}")
        if learn and not action.get("error"):
            try:
                self.memory_mgr.on_conversation_turn(session_id)
            except Exception as e:
                logger.debug(f"memory extraction skipped: {e}")
        return action


# One-shot convenience (MCP cortex_ask tool uses this).
_CORE = None


def ask(text: str, session_id: str = "mcp", source: str = "agentkit") -> str:
    return ask_full(text, session_id=session_id, source=source).get("speak", "")


def ask_full(text: str, session_id: str = "mcp", source: str = "agentkit") -> dict:
    global _CORE
    if _CORE is None:
        _CORE = CortexCore()
    return _CORE.ask(text, session_id=session_id, source=source)
