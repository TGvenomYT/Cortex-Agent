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

Safety: whole-system-destructive commands (safety.is_catastrophic) are refused
outright and surfaced to the user; everything else runs, because the user
explicitly asked Cortex to complete the job.
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("cortex.core")

_FINAL_SYNTH = (
    "You just worked on a task for {who}. Here's a log of what you did and found.\n"
    "Give them the result in a couple of warm, plain sentences — no jargon, no raw "
    "output, lead with the outcome. If the task is done, say so; if something blocked "
    "you, say what and suggest the next move.\n\nWORK LOG:\n{log}\n\n"
    'Respond as JSON: {{"speak": "..."}}'
)


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
        self.max_steps = self.config.get("agent", {}).get("max_steps", 6)
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
                return (f"[CONFIGURED SERVER] '{s.get('label','server')}' — host {s['host']}, "
                        f"ssh user {s.get('user','')}. For 'my server'/'the server', use: "
                        f"ssh -i ~/.ssh/id_ed25519_myserver {s.get('user','')}@{s['host']} \"<cmd>\".")
        except Exception:
            pass
        return None

    def _render_chart(self, code: str) -> str:
        import uuid
        import cortex.llm.viz as viz
        art = os.path.expanduser("~/.cortex/artifacts")
        os.makedirs(art, exist_ok=True)
        fn = uuid.uuid4().hex[:12] + ".png"
        ok, err = viz.render_matplotlib(code, os.path.join(art, fn))
        if ok:
            return "/artifact/" + fn
        logger.warning(f"chart render failed: {err}")
        return None

    _CHART_WORDS = ("graph", "plot", "chart", "draw", "visuali", "diagram", "curve", "histogram")

    def _wants_chart(self, text: str) -> bool:
        t = (text or "").lower()
        return any(w in t for w in self._CHART_WORDS)

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
                import re
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
                return ("[FROM THE USER'S UPLOADED FILES — use/cite when relevant]\n" +
                        "\n---\n".join(f"({h['source']}) {h['chunk']}" for h in hits))
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
        return TOOLS + extra

    # ── public: non-streaming (consumes the stream) ────────────────────────
    def ask(self, text: str, session_id: str = "cli", source: str = "cli") -> dict:
        final = {"action": "answer", "speak": "All done.", "steps": 0}
        for ev in self.ask_stream(text, session_id=session_id, source=source):
            if ev.get("type") == "final":
                final = ev["action"]
        return final

    # ── public: streaming — yields live events so a UI/CLI can show thinking ─
    def ask_stream(self, text: str, session_id: str = "cli", source: str = "cli"):
        """
        Generator of events as Cortex works:
          {"type":"thinking","text":...}      model reasoning before a step
          {"type":"tool","name":...,"detail":...}  a tool is about to run
          {"type":"observation","text":...}   what the tool returned (short)
          {"type":"answer","text":...}         the final plain-language reply
          {"type":"final","action":{...}}      full action dict (for ask())
        """
        from cortex.llm.tools import tool_call_to_action_data
        tools_list = self._tools()

        yield {"type": "status", "text": "thinking…"}
        try:
            memory_str = self.memory_mgr.format_for_prompt(limit=15)
        except Exception:
            memory_str = None
        extra = "\n\n".join(x for x in [self._server_context(), self._rag_context(text)] if x) or None
        messages = self.context_reader.build_messages(
            user_text=text, session_id=session_id, memory_str=memory_str, source=source,
            extra_context=extra,
        )

        work_log: list[str] = []
        last_action = {"action": "answer"}
        final_speak = None
        asked, ask_options = False, []
        chart_shown = False

        for step in range(1, self.max_steps + 1):
            try:
                result = self.llm.complete_with_tools(messages, tools=tools_list)
            except Exception as e:
                logger.error(f"LLM unavailable: {e}")
                final_speak = (
                    "I can't reach a model right now. Either start the local engine "
                    "(install & run Ollama, then `ollama pull gemma3:12b`), or add a cloud "
                    "API key under **Settings → AI Providers** (Claude / OpenAI / Gemini) and "
                    "pick that model from the selector at the top-left.")
                break

            think = (result.get("thinking") or result.get("content") or "").strip()
            tool_calls = result.get("tool_calls") or []

            if not tool_calls:
                final_speak = result.get("content") or "All done."
                break

            if think:
                yield {"type": "thinking", "text": think[:600]}

            messages.append({"role": "assistant",
                             "content": result.get("content", "") or "",
                             "tool_calls": tool_calls})

            stop = False
            for tc in tool_calls:
                _tcid = tc.get("id") or ""   # OpenAI/Anthropic need tool results to reference this
                _nm = (tc.get("function") or {}).get("name", "")
                if _nm.startswith("mcp__"):
                    action = {"action": "mcp", "mcp_name": _nm,
                              "mcp_args": (tc.get("function") or {}).get("arguments") or {}}
                else:
                    action = tool_call_to_action_data(tc)
                last_action = action
                if action.get("action") == "ask":   # the model needs the user to decide
                    final_speak = action.get("question") or "Quick question —"
                    ask_options = action.get("options") or []
                    asked = stop = True
                    break
                if action.get("action") == "chart":  # render a plot inline
                    src = self._render_chart(action.get("code", ""))
                    if src:
                        chart_shown = True
                        yield {"type": "image", "src": src, "caption": action.get("caption", "")}
                    obs = "chart rendered and shown to the user" if src else "chart render failed"
                    work_log.append(f"[chart] {obs}")
                    messages.append({"role": "tool", "tool_call_id": _tcid, "content": obs})
                    continue
                if action.get("action") == "image_search":  # web images inline
                    items = self._image_search(action.get("query", ""))
                    if items:
                        yield {"type": "images", "items": items[:8], "query": action.get("query", "")}
                    obs = (f"showed {len(items)} web images for '{action.get('query')}'"
                           if items else "no images found")
                    work_log.append(f"[image_search] {obs}")
                    messages.append({"role": "tool", "tool_call_id": _tcid, "content": obs})
                    continue
                yield {"type": "tool", "name": action.get("action", "?"),
                       "detail": self._tool_detail(action)}
                obs = self._execute_tool(action, source)
                work_log.append(f"[{action.get('action')}] {obs}")
                yield {"type": "observation", "text": obs[:400]}
                messages.append({"role": "tool", "tool_call_id": _tcid, "content": obs[:3000]})
            if stop:
                break
        else:
            final_speak = None

        # Guaranteed-chart fallback: the user asked for a plot but the model never
        # produced one (it narrated "here's the graph" without calling make_chart —
        # common on weaker/cloud tool-callers). Generate the code directly and render
        # it, and NEVER let the reply claim a graph that doesn't exist.
        if not asked and self._wants_chart(text) and not chart_shown:
            yield {"type": "status", "text": "rendering the chart…"}
            code = self._chart_code_for(text)
            src = self._render_chart(code) if code else None
            if src:
                chart_shown = True
                yield {"type": "image", "src": src, "caption": ""}
                if not final_speak or "graph" in (final_speak or "").lower() or \
                        "plot" in (final_speak or "").lower() or "chart" in (final_speak or "").lower():
                    final_speak = final_speak or "Here's the graph."
                last_action = {"action": "chart"}
            else:
                final_speak = ("I couldn't render that plot — the code my model produced didn't "
                               "run. Tell me the exact equation or range and I'll try again.")

        if final_speak is None:
            yield {"type": "status", "text": "wrapping up…"}
            final_speak = self._final_summary(work_log)

        last_action["speak"] = final_speak
        last_action.setdefault("action", "answer")
        last_action["steps"] = len(work_log)
        self._finish(session_id, text, last_action)
        yield {"type": "answer", "text": final_speak}
        if asked:
            yield {"type": "ask", "options": ask_options}
        else:
            fu = self._followups(text, final_speak)
            if fu:
                yield {"type": "followups", "items": fu}
        yield {"type": "final", "action": last_action}

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
        if kind == "mcp":
            return action.get("mcp_name", "").replace("mcp__", "").replace("__", " · ")
        if kind == "chart":
            return action.get("caption", "") or "plot"
        if kind == "image_search":
            return action.get("query", "")
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
                return self._run_shell(
                    f"cd {action.get('run_project_path','.')} && "
                    f"{action.get('run_project_command','')}", source)
            if kind == "mcp":
                return self.mcp.call(action.get("mcp_name", ""), action.get("mcp_args") or {}) if self.mcp else "MCP not available"
            if kind == "email_check":
                return self._email_check()
            if kind == "email_search":
                return self._email_search(action.get("email_query", ""))
            if kind == "email_send":
                return self._email_send(action)
            if kind == "remind":
                return self._handle_reminder(action.get("reminder") or {})
            if kind == "schedule_task":
                return self._handle_schedule_task(action)
            if kind == "credential_prompt":
                return ("Needs a login I don't have stored. Ask the user for the "
                        f"{action.get('credential_prompt_site','site')} credentials next time you talk.")
            return "Nothing to do for that step."
        except Exception as e:
            return f"that step hit an error: {e}"

    def _run_shell(self, command: str, source: str) -> str:
        from cortex.core.safety import is_catastrophic
        if not command:
            return "no command given"
        bad, why = is_catastrophic(command)
        if bad:
            logger.warning(f"refused catastrophic command: {command}")
            return f"REFUSED — {why}"
        res = self.executor.execute(command=command, source="cortex-core")
        code = res.get("exit_code")
        out = (res.get("output") or res.get("stdout") or "").strip()
        err = (res.get("stderr") or "").strip()
        tail = out[-1500:] if out else (err[-800:] if err else "(no output)")
        return f"ran `{command}` -> exit {code}; {tail}"

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

    def _handle_schedule_task(self, action: dict) -> str:
        import cortex.core.tasks as tasks
        op = action.get("task_op", "add")
        if op == "list":
            ts = tasks.list_tasks()
            if not ts:
                return "no recurring tasks scheduled"
            return "scheduled tasks: " + "; ".join(
                f"[{t['id']}] {t['desc']} — {t.get('human', t['schedule'])} → {t.get('deliver','notify')}"
                for t in ts)
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

    def _finish(self, session_id: str, user_text: str, action: dict) -> dict:
        try:
            self.db.save_conversation(session_id, "user", user_text)
            self.db.save_conversation(session_id, "assistant", action.get("speak", ""))
        except Exception:
            pass
        try:
            self.db.log_experience(session_id=session_id, user_text=user_text,
                                   action=action.get("action", "answer"),
                                   action_payload={k: v for k, v in action.items() if k != "speak"},
                                   speak=action.get("speak"))
        except Exception:
            pass
        return action


# One-shot convenience (MCP cortex_ask tool uses this).
_CORE = None


def ask(text: str, session_id: str = "mcp", source: str = "cli") -> str:
    global _CORE
    if _CORE is None:
        _CORE = CortexCore()
    return _CORE.ask(text, session_id=session_id, source=source).get("speak", "")
