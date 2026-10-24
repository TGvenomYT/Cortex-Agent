#!/usr/bin/env python3
"""
gui_server.py — Cortex web GUI, laid out exactly like Gemini (dark): recent-chat
sidebar on the left, chat on the right with right-aligned user bubbles and plain
markdown responses, and the Gemini-style input pill. Everything is real (real
chat history from the DB, real model name, real profile) — no placeholders.

Streams the agent's live thinking over SSE; voice is on-device MLX (mic →
mlx-whisper, answers → kokoro-mlx).

Run: python3 gui_server.py   (or: cx gui)  →  http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import json
import logging

os.chdir(__import__("cortex").ROOT)

import httpx
from flask import Flask, request, Response, jsonify, stream_with_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("cortex.gui")

app = Flask(__name__)
_CORE = None
VOICE_PORT = int(os.environ.get("CORTEX_VOICE_PORT", "8005"))
VOICE_URL = f"http://127.0.0.1:{VOICE_PORT}"
_voice_proc = None


def core():
    global _CORE
    if _CORE is None:
        from cortex.core.cortex_core import CortexCore
        _CORE = CortexCore()
    return _CORE


def _model_label() -> str:
    try:
        m = core().config["llm"]["ollama"]["model"]
    except Exception:
        return "Cortex"
    m = m.split("/")[-1]
    if "gemma4-agent" in m:
        return "Gemma 12B"
    if "12b" in m:
        return "Gemma 12B"
    return m


def ensure_voice():
    global _voice_proc
    try:
        httpx.get(VOICE_URL + "/health", timeout=2); return True
    except Exception:
        pass
    import cortex.voice.voice_mode as voice_mode
    _voice_proc = voice_mode.ensure_server()
    try:
        httpx.get(VOICE_URL + "/health", timeout=2); return True
    except Exception:
        return False


@app.route("/")
def index():
    from cortex.core.context_reader import ContextReader
    r = Response(PAGE.replace("__OWNER__", ContextReader.owner_name())
                     .replace("__MODEL__", _model_label()), mimetype="text/html")
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"  # always serve fresh app
    return r


@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "Cortex", "short_name": "Cortex", "start_url": "/",
        "display": "standalone", "background_color": "#131314", "theme_color": "#131314",
        "description": "Your on-device AI — a brother in the machine.",
        "icons": [
            {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    })


@app.route("/artifact/<name>")
def artifact(name):
    path = os.path.join(os.path.expanduser("~/.cortex/artifacts"), os.path.basename(name))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    return Response(b"", status=404)


@app.route("/icon-<int:sz>.png")
def icon(sz):
    path = os.path.join(__import__("cortex").ROOT, "assets", f"icon-{sz}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    return Response(b"", status=404)


def _asset(name, mime):
    path = os.path.join(__import__("cortex").ROOT, "assets", name)
    if os.path.exists(path):
        mode = "r" if mime.endswith("svg+xml") else "rb"
        with open(path, mode) as f:
            return Response(f.read(), mimetype=mime)
    return Response(b"", status=404)


@app.route("/favicon.svg")
def favicon_svg():
    return _asset("favicon.svg", "image/svg+xml")


@app.route("/favicon.ico")
def favicon_ico():
    # browsers auto-request /favicon.ico; serve the 32px PNG (widely accepted)
    return _asset("favicon-32.png", "image/png")


@app.route("/favicon-32.png")
def favicon_32():
    return _asset("favicon-32.png", "image/png")


@app.route("/apple-touch-icon.png")
def apple_touch_icon():
    return _asset("apple-touch-icon.png", "image/png")


@app.route("/sw.js")
def sw():
    # SELF-DESTRUCTING service worker. A cache-first SW pinned stale app code in
    # installed PWAs and hid every fix. Cortex always needs the local server
    # running anyway, so offline shell-caching buys nothing — this SW deletes all
    # caches, unregisters itself, and reloads open windows so a stuck PWA heals in
    # one load and is never controlled by a SW again (the page no longer registers one).
    js = """
self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil((async()=>{
  const keys=await caches.keys();
  await Promise.all(keys.map(k=>caches.delete(k)));
  await self.clients.claim();
  await self.registration.unregister();
  const cs=await self.clients.matchAll({type:'window'});
  cs.forEach(c=>c.navigate(c.url));
})()));
"""
    r = Response(js, mimetype="application/javascript")
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"  # always fetch the current SW
    return r


@app.route("/models")
def models():
    """Local Ollama models + cloud models for any provider with an API key."""
    import cortex.llm.llm as llm_mod
    out = {"current": "", "models": []}
    try:
        out["current"] = core().llm.primary.model
    except Exception:
        pass
    local = []
    try:
        base = core().config["llm"]["ollama"].get("base_url", "http://localhost:11434")
        data = httpx.get(base + "/api/tags", timeout=4).json()
        local = sorted(m["name"] for m in data.get("models", []))
    except Exception as e:
        logger.warning(f"/models: {e}")
    cloud = []
    for prov, has in llm_mod.configured_providers().items():
        if has:
            cloud += llm_mod.CLOUD_MODELS.get(prov, [])
    out["models"] = local + cloud
    out["local"] = local
    out["cloud"] = cloud
    return jsonify(out)


@app.route("/set_model", methods=["POST"])
def set_model():
    """Switch the live model — rebuilds the backend so cloud providers work too."""
    import cortex.llm.llm as llm_mod
    m = (request.get_json(silent=True) or {}).get("model", "").strip()
    if not m:
        return jsonify({"ok": False}), 400
    try:
        c = core()
        c.llm.primary = llm_mod.backend_for_model(m, c.config.get("llm", {}))
        logger.info(f"model switched to {m} ({llm_mod.provider_for_model(m)})")
        return jsonify({"ok": True, "current": m})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/keys")
def keys_status():
    """Which cloud providers have a key set (never returns the keys themselves)."""
    import cortex.llm.llm as llm_mod
    return jsonify({"providers": llm_mod.configured_providers()})


@app.route("/keys/set", methods=["POST"])
def keys_set():
    import cortex.llm.llm as llm_mod
    d = request.get_json(force=True) or {}
    prov, key = d.get("provider", ""), (d.get("key", "") or "").strip()
    if prov not in llm_mod.CLOUD_MODELS or not key:
        return jsonify({"ok": False, "error": "provider and key required"}), 400
    ok = llm_mod.set_key(prov, key)
    return jsonify({"ok": ok})


@app.route("/keys/remove", methods=["POST"])
def keys_remove():
    import cortex.llm.llm as llm_mod
    prov = (request.get_json(force=True) or {}).get("provider", "")
    if prov in llm_mod.CLOUD_MODELS:
        llm_mod.delete_key(prov)
    return jsonify({"ok": True})


@app.route("/settings", methods=["GET", "POST"])
def settings():
    import cortex.storage.settings_store as settings_store
    if request.method == "POST":
        return jsonify(settings_store.save(request.get_json(silent=True) or {}))
    return jsonify(settings_store.load())


@app.route("/avatar", methods=["GET", "POST"])
def avatar():
    import cortex.storage.settings_store as settings_store
    if request.method == "POST":
        f = request.files.get("avatar")
        if not f:
            return jsonify({"ok": False}), 400
        settings_store.save_avatar(f.read())
        return jsonify({"ok": True})
    if os.path.exists(settings_store.AVATAR_FILE):
        with open(settings_store.AVATAR_FILE, "rb") as fh:
            return Response(fh.read(), mimetype="image/png")
    return Response(b"", status=404)


@app.route("/mail/accounts")
def mail_accounts():
    import cortex.integrations.mail_store as mail_store
    return jsonify({"accounts": mail_store.list_accounts()})


@app.route("/mail/add", methods=["POST"])
def mail_add():
    import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store
    b = request.get_json(silent=True) or {}
    addr = (b.get("email") or "").strip()
    pw = (b.get("password") or "").replace(" ", "")  # app pw shown with spaces
    if not addr or not pw:
        return jsonify({"ok": False, "error": "email and app password required"}), 400
    ok, err = mail.test_login(addr, pw)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400
    mail_store.add(addr, pw, b.get("label", ""))
    return jsonify({"ok": True})


@app.route("/mail/remove", methods=["POST"])
def mail_remove():
    import cortex.integrations.mail_store as mail_store
    addr = (request.get_json(silent=True) or {}).get("email", "")
    mail_store.remove(addr)
    return jsonify({"ok": True})


@app.route("/mail/unread")
def mail_unread():
    """Unread counts per account (slow — each hits IMAP). Loaded on demand."""
    import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store
    out = []
    for a in mail_store.list_accounts():
        pw = mail_store.password(a["email"])
        n = mail.unread_count(a["email"], pw) if pw else -1
        out.append({"email": a["email"], "label": a.get("label", ""), "unread": n})
    return jsonify({"accounts": out})


@app.route("/mcp/servers")
def mcp_servers():
    import cortex.storage.settings_store as settings_store
    return jsonify({"servers": settings_store.load().get("mcp_servers", [])})


@app.route("/mcp/status")
def mcp_status():
    try:
        return jsonify({"status": core().mcp.status() if core().mcp else []})
    except Exception as e:
        return jsonify({"status": [], "error": str(e)})


def _reload_mcp():
    import cortex.storage.settings_store as settings_store
    try:
        from cortex.integrations.mcp_client import MCPManager
        c = core()
        if c.mcp:
            c.mcp.stop()
        c.mcp = MCPManager(settings_store.load().get("mcp_servers", []))
    except Exception as e:
        logger.warning(f"mcp reload: {e}")


@app.route("/mcp/add", methods=["POST"])
def mcp_add():
    import cortex.storage.settings_store as settings_store
    b = request.get_json(silent=True) or {}
    name, cmd = (b.get("name") or "").strip(), (b.get("command") or "").strip()
    if not name or not cmd:
        return jsonify({"ok": False, "error": "name and command required"}), 400
    servers = [s for s in settings_store.load().get("mcp_servers", []) if s.get("name") != name]
    servers.append({"name": name, "command": cmd})
    settings_store.save({"mcp_servers": servers})
    _reload_mcp()
    return jsonify({"ok": True})


@app.route("/mcp/remove", methods=["POST"])
def mcp_remove():
    import cortex.storage.settings_store as settings_store
    name = (request.get_json(silent=True) or {}).get("name", "")
    servers = [s for s in settings_store.load().get("mcp_servers", []) if s.get("name") != name]
    settings_store.save({"mcp_servers": servers})
    _reload_mcp()
    return jsonify({"ok": True})


@app.route("/devices")
def devices():
    """Connected devices + live hardware usage."""
    import cortex.storage.settings_store as settings_store
    out = []
    # This Mac — from the local collectors' context
    mac = {"label": "This Mac", "kind": "laptop", "status": "online",
           "cpu": None, "mem": None, "disk": None, "detail": ""}
    try:
        seen = set()
        for c in core().db.get_context(categories=["system"], max_age_minutes=180):
            if c["key"] in seen:
                continue
            seen.add(c["key"])
            v = c.get("value") or {}
            if c["key"] == "cpu" and isinstance(v, dict):
                load = float(v.get("load_1min") or 0); cores = float(v.get("logical_cores") or 1)
                mac["cpu"] = round(min(100.0, load / cores * 100))
            elif c["key"] == "memory" and isinstance(v, dict):
                u = v.get("used_mb"); t = v.get("total_mb")
                if u and t:
                    mac["mem"] = round(u / t * 100); mac["detail"] = f"{round(u/1024,1)}/{round(t/1024,1)} GB RAM"
            elif c["key"] == "disk" and isinstance(v, dict):
                vols = v.get("all_volumes") or []
                if vols:
                    mac["disk"] = str(vols[0].get("capacity", ""))
    except Exception as e:
        logger.warning(f"/devices mac: {e}")
    out.append(mac)
    return jsonify({"devices": out})   # fast: local only; server loaded via /device_server


@app.route("/device_server")
def device_server():
    """The server's live status (slow — devops over Tailscale). Loaded separately
    so it never blocks the local device card."""
    import cortex.storage.settings_store as settings_store
    s = settings_store.load().get("server", {})
    srv = {"label": s.get("label") or "myserver", "kind": "server", "status": "offline",
           "cpu": None, "mem": None, "disk": None, "detail": "unreachable (Tailscale off?)"}
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.expanduser("~/agent-tools"))
        from agentkit import registry
        st = registry.call("devops_status", {})
        host = (st or {}).get("host", {})
        srv["status"] = "online"
        srv["cpu"] = round(host.get("cpu_pct")) if host.get("cpu_pct") is not None else None
        srv["mem"] = round(host.get("memory_pct")) if host.get("memory_pct") is not None else None
        srv["detail"] = f"{len((st or {}).get('containers', []))} containers"
    except Exception as e:
        logger.info(f"/device_server offline: {e}")
    return jsonify({"device": srv})


_JOB_META = {
    "autonomy": ("Autonomy", "every 30 min · thinks & acts on its own"),
    "reminders": ("Reminders", "every 60 s · fires what's due"),
    "automation": ("Automation rules", "every 2 min · threshold rules"),
    "sync": ("Cloud sync", "every 30 min · pushes to Supabase"),
    "report-morning": ("Morning briefing", "daily 09:00"),
    "report-evening": ("Evening wrap", "daily 18:00"),
}


@app.route("/jobs")
def jobs():
    """What Cortex is running right now: scheduled background jobs + active goals."""
    import subprocess
    out = {"scheduled": [], "goals": []}
    try:
        res = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=5)
        for line in res.stdout.splitlines():
            if "com.cortex." not in line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            pid, status, label = parts[0].strip(), parts[1].strip(), parts[2].strip()
            key = label.replace("com.cortex.", "")
            if key.startswith("collector.") or key.startswith("task."):
                continue  # collectors are plumbing; user tasks listed separately below
            name, desc = _JOB_META.get(key, (key, ""))
            out["scheduled"].append({
                "key": key, "name": name, "desc": desc,
                "running": pid not in ("-", ""), "ok": status in ("0", "-"),
            })
    except Exception as e:
        logger.warning(f"/jobs: {e}")
    try:
        out["goals"] = [g["description"] for g in core().db.get_active_goals(limit=8)]
    except Exception:
        pass
    try:
        out["reminders"] = [{"text": r["text"], "due": str(r["due_at"])}
                            for r in core().db.get_all_reminders(include_fired=False)[:8]]
    except Exception:
        out["reminders"] = []
    try:
        import cortex.core.tasks as tasks_mod
        out["tasks"] = tasks_mod.list_tasks()
    except Exception:
        out["tasks"] = []
    return jsonify(out)


@app.route("/tasks/add", methods=["POST"])
def tasks_add():
    import cortex.core.tasks as tasks_mod
    d = request.get_json(force=True) or {}
    if not d.get("prompt") or not d.get("schedule"):
        return jsonify({"error": "prompt and schedule required"}), 400
    t = tasks_mod.add_task(desc=d.get("desc", ""), prompt=d["prompt"],
                           schedule=d["schedule"], deliver=d.get("deliver", "notify"))
    return jsonify(t)


@app.route("/tasks/remove", methods=["POST"])
def tasks_remove():
    import cortex.core.tasks as tasks_mod
    tid = (request.get_json(force=True) or {}).get("id", "")
    tasks_mod.remove_task(tid)
    return jsonify({"ok": True})


@app.route("/nudges")
def nudges_list():
    """Proactive notices Cortex has surfaced (for the top-bar bell)."""
    try:
        items = core().db.get_nudges(limit=20)
        unread = core().db.count_unread_nudges()
    except Exception as e:
        logger.warning(f"/nudges: {e}")
        items, unread = [], 0
    return jsonify({"nudges": [{"id": n["id"], "title": n["title"], "body": n.get("body", ""),
                               "kind": n.get("kind", "info"), "read": bool(n["read"]),
                               "when": str(n["created_at"])} for n in items],
                    "unread": unread})


@app.route("/nudges/read", methods=["POST"])
def nudges_read():
    nid = (request.get_json(force=True) or {}).get("id")
    try:
        core().db.mark_nudges_read(nid)
    except Exception as e:
        logger.warning(f"/nudges/read: {e}")
    return jsonify({"ok": True})


@app.route("/nudges/clear", methods=["POST"])
def nudges_clear():
    try:
        core().db.clear_nudges()
    except Exception as e:
        logger.warning(f"/nudges/clear: {e}")
    return jsonify({"ok": True})


@app.route("/chats")
def chats():
    try:
        return jsonify({"chats": core().db.get_chat_sessions(limit=40)})
    except Exception as e:
        logger.warning(f"/chats: {e}")
        return jsonify({"chats": []})


@app.route("/history")
def history():
    session = request.args.get("session", "")
    try:
        msgs = core().db.get_conversations(session_id=session, limit=80)
    except Exception:
        msgs = []
    out = []
    for m in msgs:
        content = m.get("content", "")
        if m.get("role") == "assistant" and content.strip().startswith("{"):
            try:
                content = json.loads(content).get("speak", content)
            except Exception:
                pass
        out.append({"role": m.get("role"), "content": content})
    return jsonify({"messages": out})


@app.route("/files/upload", methods=["POST"])
def files_upload():
    import cortex.storage.rag as rag
    updir = os.path.expanduser("~/.cortex/uploads")
    os.makedirs(updir, exist_ok=True)
    results = []
    for f in request.files.getlist("files"):
        safe = os.path.basename(f.filename or "file")
        path = os.path.join(updir, safe)
        try:
            f.save(path)
            src, n = rag.ingest(core().db, path, source=safe)
            results.append({"name": src, "chunks": n})
        except Exception as e:
            results.append({"name": safe, "error": str(e)})
    return jsonify({"files": results})


@app.route("/files")
def files_list():
    try:
        return jsonify({"files": core().db.list_documents()})
    except Exception:
        return jsonify({"files": []})


@app.route("/stream")
def stream():
    q = request.args.get("q", "").strip()
    session = request.args.get("session", "gui")
    if not q:
        return Response("data: " + json.dumps({"type": "answer", "text": "Ask me something."}) + "\n\n",
                        mimetype="text/event-stream")

    @stream_with_context
    def gen():
        try:
            for ev in core().ask_stream(q, session_id=session, source="gui"):
                yield "data: " + json.dumps(ev) + "\n\n"
        except Exception as e:
            logger.error(f"stream error: {e}")
            yield "data: " + json.dumps({"type": "answer", "text": f"Something went wrong: {e}"}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/voice/stt", methods=["POST"])
def voice_stt():
    if not ensure_voice():
        return jsonify({"text": "", "error": "voice server unavailable"}), 503
    f = request.files.get("audio")
    if not f:
        return jsonify({"text": ""}), 400
    try:
        r = httpx.post(VOICE_URL + "/v1/audio/transcriptions",
                       files={"file": ("speech.webm", f.read(), "audio/webm")},
                       data={"model": "whisper-1", "language": "en"}, timeout=60)
        return jsonify({"text": (r.json().get("text") or "").strip()})
    except Exception as e:
        return jsonify({"text": "", "error": str(e)}), 502


@app.route("/voice/tts", methods=["POST"])
def voice_tts():
    if not ensure_voice():
        return Response(b"", status=503)
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    import cortex.storage.settings_store as settings_store
    voice = body.get("voice") or settings_store.load().get("voice") or os.environ.get("CORTEX_TTS_VOICE", "bm_daniel")
    try:
        r = httpx.post(VOICE_URL + "/v1/audio/speech",
                       json={"model": "kokoro", "input": text[:1200], "voice": voice,
                             "response_format": "wav"}, timeout=120)
        return Response(r.content, mimetype="audio/wav")
    except Exception as e:
        logger.warning(f"tts: {e}")
        return Response(b"", status=502)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cortex</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="shortcut icon" href="/favicon.ico">
<link rel="manifest" href="/manifest.json"><meta name="theme-color" content="#131314">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-title" content="Cortex">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"><link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>
:root{--bg:#131314;--side:#1e1f20;--hover:#333537;--sel:#1b3a5b;--txt:#e3e3e3;
 --mut:#c4c7c5;--dim:#9aa0a6;--line:#444746;--bubble:#1e1f20;--pill:#1e1f20;--blue:#8ab4f8;--card:#17181a;
 --grad:linear-gradient(74deg,#4285f4 0%,#9b72cb 48%,#d96570 100%)}
html[data-theme=light]{--bg:#ffffff;--side:#f0f4f9;--hover:#e4e9f1;--sel:#d3e3fd;--txt:#1f1f1f;
 --mut:#444746;--dim:#5f6368;--line:#c4c7c5;--bubble:#f0f4f9;--pill:#f0f4f9;--blue:#0b57d0;--card:#f7f9fc}
html,body{background:var(--bg)}
/* circular-reveal theme switch (View Transitions API) */
::view-transition-old(root),::view-transition-new(root){animation:none;mix-blend-mode:normal}
::view-transition-old(root){z-index:0}::view-transition-new(root){z-index:1}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--txt);display:flex;
 font:15px/1.6 "Google Sans",Roboto,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
/* sidebar */
#side{width:300px;background:var(--side);display:flex;flex-direction:column;flex:0 0 auto;overflow:hidden;padding:10px 10px}
.shead{display:flex;align-items:center;gap:10px;padding:8px 10px 12px}
.logo{display:flex;align-items:center;gap:10px;font-size:20px;color:var(--mut);font-weight:500}
.spark{font-size:22px;line-height:1;flex:0 0 auto;background:var(--grad);
 -webkit-background-clip:text;background-clip:text;color:transparent}
.row{display:flex;align-items:center;gap:14px;border-radius:999px;padding:11px 14px;cursor:pointer;color:var(--mut);white-space:nowrap}
.row:hover{background:var(--hover)}.row .ic{width:20px;text-align:center;font-size:17px;flex:0 0 auto}
#new{color:var(--txt);margin-bottom:6px}
#srch{margin:4px 6px 8px;display:flex;align-items:center;gap:8px;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:8px 14px}
#srch input{flex:1;background:transparent;border:0;outline:none;color:var(--txt);font:inherit;font-size:13px}
.sect{color:var(--dim);font-size:13px;padding:10px 16px 4px}
#list{flex:1;overflow:auto;margin:0 -4px;padding:0 4px}
.chat{border-radius:999px;padding:9px 14px;cursor:pointer;color:var(--mut);white-space:nowrap;
 overflow:hidden;text-overflow:ellipsis;font-size:14px}
.chat:hover{background:var(--hover)}.chat.sel{background:var(--sel);color:var(--blue)}
.prof{display:flex;align-items:center;gap:12px;padding:10px 10px;margin-top:6px;border-top:1px solid #2a2c2e}
.pav{width:30px;height:30px;border-radius:50%;background:var(--grad);flex:0 0 auto}
.pname{font-size:14px}.psub{font-size:12px;color:var(--dim)}
/* model dropdown */
.mbtn{color:var(--mut);font-size:15px;padding:7px 12px;border-radius:999px;cursor:pointer;background:transparent;border:0;font:inherit}
.mbtn:hover{background:var(--hover)}
#menu{position:absolute;top:48px;left:16px;background:var(--card);border:1px solid var(--line);border-radius:16px;
 padding:6px;min-width:240px;max-width:340px;max-height:360px;overflow:auto;z-index:60;display:none;box-shadow:0 10px 36px #000a}
#menu.show{display:block}
.mi{padding:9px 12px;border-radius:10px;cursor:pointer;font-size:14px;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mi:hover{background:var(--hover)}.mi .c{color:var(--blue);float:right}
/* main */
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#top{display:flex;align-items:center;padding:8px 16px;gap:6px;position:relative;z-index:10}
#scroll{flex:1;overflow:auto}
#wrap{max-width:820px;margin:0 auto;padding:6px 24px 30px;width:100%}
#hello{max-width:820px;margin:22vh auto 0;padding:0 24px;font-size:44px;font-weight:500;
 background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.turn{margin:26px 0}
.urow{display:flex;justify-content:flex-end}
.ubub{background:var(--bubble);border-radius:22px;padding:11px 18px;max-width:78%}
.resp{margin-top:18px}
.trace{border:1px solid var(--line);background:var(--card);border-radius:16px;margin:0 0 12px;overflow:hidden}
.trace summary{cursor:pointer;padding:8px 14px;color:var(--dim);font-size:13px;list-style:none;display:flex;align-items:center;gap:9px}
.trace summary::-webkit-details-marker{display:none}
.trace summary:hover{background:#1d1e20}
.trace .body{padding:4px 14px 9px;border-top:1px solid var(--line)}
.step{display:flex;gap:8px;padding:3px 0;font-size:13px}.step .k{color:var(--blue);min-width:72px;font-weight:600}
.step .d,.obs{color:var(--dim);font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}
.obs{padding:1px 0 5px 80px}
.ans h1,.ans h2,.ans h3{line-height:1.3;margin:.7em 0 .35em}.ans h1{font-size:1.3em}.ans h2{font-size:1.15em}.ans h3{font-size:1.04em}
.ans code{background:#0b0d11;border:1px solid var(--line);border-radius:5px;padding:.08em .35em;font-family:ui-monospace,Menlo,monospace;font-size:.9em}
.ans pre{background:#0b0d11;border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto}.ans pre code{border:0;padding:0}
.ans ul,.ans ol{padding-left:1.3em}.ans a{color:var(--blue)}
.spin{width:13px;height:13px;border:2px solid var(--line);border-top-color:var(--blue);border-radius:50%;display:inline-block;animation:s .8s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes s{to{transform:rotate(360deg)}}
/* input pill */
#dock{padding:8px 24px 18px}
#attached{max-width:820px;margin:0 auto 8px;display:flex;flex-wrap:wrap;gap:6px}
#form{max-width:820px;margin:0 auto;display:flex;align-items:flex-end;gap:6px;background:var(--pill);
 border:1px solid var(--line);border-radius:26px;padding:7px 10px 7px 20px}
#q{flex:1;background:transparent;border:0;outline:none;color:var(--txt);font:inherit;padding:9px 2px;resize:none;max-height:150px}
.model-sel{color:var(--mut);font-size:13px;padding:8px 6px;white-space:nowrap;align-self:center}
.rbtn{width:40px;height:40px;border-radius:50%;border:0;background:transparent;color:var(--mut);cursor:pointer;display:flex;align-items:center;justify-content:center}
.rbtn:hover{background:var(--hover)}.rbtn.on{color:var(--blue)}.rbtn.rec{color:#f28b82;animation:p 1s infinite}
@keyframes p{50%{background:#f28b8222}}
.bellwrap{position:relative}
.badge{position:absolute;top:4px;right:4px;min-width:16px;height:16px;padding:0 4px;border-radius:9px;background:#ea4335;color:#fff;font-size:10px;font-weight:700;line-height:16px;text-align:center;display:none}
.badge.show{display:block}
#nudgepanel{position:absolute;top:46px;right:0;width:330px;max-height:420px;overflow-y:auto;background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.35);padding:8px;display:none;z-index:50}
#nudgepanel.show{display:block}
.nhead{display:flex;align-items:center;justify-content:space-between;padding:8px 10px 6px}
.nhead b{font-size:14px}.nhead span{font-size:12px;color:var(--blue);cursor:pointer}
.nitem{padding:10px 12px;border-radius:12px;cursor:default}
.nitem:hover{background:var(--hover)}
.nitem.un{background:var(--hover)}
.nitem .nt{font-size:13px;font-weight:600;display:flex;gap:6px;align-items:center}
.nitem .nb{font-size:12px;color:var(--mut);margin-top:3px;line-height:1.4}
.nitem .nw{font-size:11px;color:var(--dim);margin-top:4px}
.ndot{width:7px;height:7px;border-radius:50%;flex:none}
.ndot.info{background:var(--blue)}.ndot.warn{background:#fbbc04}.ndot.alert{background:#ea4335}
.rbtn svg,.ic svg,#srch svg{fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;display:block}
.rbtn svg{width:20px;height:20px}.ic svg{width:20px;height:20px}
#srch svg{width:16px;height:16px;color:var(--dim);flex:0 0 auto}
#send{display:none}#send.show{display:flex}#send.show{color:#062e6f;background:#8ab4f8}#send.show:hover{background:#a8c7fa}
.chev{width:18px;height:18px;color:var(--dim);transition:transform .25s ease;margin-left:auto;flex:0 0 auto}
details.trace[open] .chev{transform:rotate(180deg)}
.jist{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.spk.muted{opacity:.5}
.foot{max-width:820px;margin:8px auto 0;text-align:center;color:var(--dim);font-size:12px}
/* ask-with-options, follow-ups, edit-prompt */
.opts{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.opt{border:1px solid var(--line);background:var(--card);color:var(--txt);border-radius:999px;padding:9px 16px;cursor:pointer;font-size:14px;transition:all .15s}
.opt:hover{border-color:var(--blue);background:var(--hover)}
.followups{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 2px}
.fu{border:1px solid var(--line);background:transparent;color:var(--mut);border-radius:999px;padding:7px 14px;cursor:pointer;font-size:13px;transition:all .15s;display:inline-flex;align-items:center;gap:7px}
.fu:hover{border-color:var(--blue);color:var(--txt)}
.fu .kbd{font-size:10px;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:0 4px}
.chartcard{margin:14px 0;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:var(--card);max-width:620px}
.chartcard img{width:100%;display:block}
.chartcap{padding:8px 14px;font-size:13px;color:var(--dim)}
.imgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:14px 0;max-width:680px}
.imgrid a{display:block;border-radius:12px;overflow:hidden;border:1px solid var(--line);aspect-ratio:1;background:var(--card)}
.imgrid img{width:100%;height:100%;object-fit:cover;display:block}
.urow{display:flex;justify-content:flex-end;align-items:center;gap:6px}
.uedit{opacity:0;cursor:pointer;color:var(--dim);transition:opacity .15s;flex:0 0 auto;padding:6px;border-radius:50%;line-height:0}
.uedit:hover{background:var(--hover);color:var(--txt)}
.urow:hover .uedit{opacity:1}.uedit svg{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:2}
/* ───── motion & the colorful thinking aura ───── */
.row,.chat,.rbtn,.mi,.mbtn,.iclk{transition:background .16s ease,color .16s ease,transform .12s ease}
#scroll{scroll-behavior:smooth}
#form{transition:border-color .2s ease,box-shadow .25s ease}
#form:focus-within{border-color:#5b8def;box-shadow:0 0 0 1px #5b8def55,0 6px 30px #4285f426}
#send.show{animation:pop .18s ease both}
.turn{animation:rise .38s cubic-bezier(.22,.61,.36,1) both}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.ubub{animation:pop .22s ease both}
.ans{animation:fade .45s ease both}@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:none}}
#menu.show{animation:pop .15s ease both;transform-origin:top left}
#hello{background-size:220% auto;animation:gshift 7s ease-in-out infinite}
@keyframes gshift{0%,100%{background-position:0% center}50%{background-position:100% center}}
/* thinking block + colorful aura: a masked rotating gradient border + soft halo */
@property --a{syntax:"<angle>";inherits:false;initial-value:0deg}
.rlive{position:relative;border-radius:16px;margin-bottom:12px}
.rlive .trace{position:relative;z-index:1;margin:0;background:var(--card);border-radius:16px}
.rlive.active .trace{border-color:transparent}
.rlive.active{animation:halo 2.8s ease-in-out infinite}
@keyframes halo{0%,100%{box-shadow:0 0 16px -8px #9b72cb55,0 0 30px -16px #4285f44d}
 50%{box-shadow:0 0 26px -6px #4285f488,0 0 46px -14px #d9657066}}
.rlive.active::before{content:"";position:absolute;inset:0;border-radius:16px;padding:1.7px;z-index:0;pointer-events:none;
 background:conic-gradient(from var(--a),#4285f4,#9b72cb,#d96570,#f6c343,#34a853,#30c5d2,#4285f4);
 -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;
 mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);mask-composite:exclude;
 animation:rot 4s linear infinite}
@keyframes rot{to{--a:360deg}}
@keyframes spin{to{transform:rotate(360deg)}}
.orb{width:15px;height:15px;border-radius:50%;display:inline-block;vertical-align:-3px;margin-right:9px;
 background:conic-gradient(from 0deg,#4285f4,#9b72cb,#d96570,#f6c343,#34a853,#4285f4);animation:spin 2.4s linear infinite;
 box-shadow:0 0 10px #9b72cb88}
.rlive:not(.active) .orb{display:none}
.shimmer{background:linear-gradient(90deg,#9aa0a6 0%,#f4f6fb 22%,#9aa0a6 44%);background-size:200% 100%;
 -webkit-background-clip:text;background-clip:text;color:transparent;animation:sh 1.7s linear infinite}
@keyframes sh{to{background-position:-200% 0}}
.obs,.step{animation:fade .3s ease both}
/* overlay + modals (settings / onboarding) */
#ov{position:fixed;inset:0;background:#0009;display:none;z-index:40;align-items:center;justify-content:center}
#ov.show{display:flex;animation:fade .2s ease}
.modal{position:relative;background:var(--bg);border:1px solid var(--line);border-radius:26px;width:min(560px,93vw);max-height:88vh;overflow:auto;padding:22px 24px;animation:pop .2s ease both}
.mx{position:absolute;top:14px;right:16px;z-index:6;width:32px;height:32px;border:0;border-radius:50%;background:transparent;color:var(--mut);cursor:pointer;display:flex;align-items:center;justify-content:center}
.mx:hover{background:var(--hover)}.mx svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round}
.mhead{display:flex;align-items:center;justify-content:space-between}
.modal h2{margin:.1em 0;font-size:22px;font-weight:600;color:var(--txt)}
.modal .sub{color:var(--dim);font-size:13px;margin:2px 0 8px}
.xbtn{cursor:pointer;color:var(--mut);border-radius:50%;padding:6px;line-height:0;display:flex}.xbtn:hover{background:var(--hover)}
.xbtn svg{width:20px;height:20px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round}
.sgroup{margin:14px 0 2px;padding-top:14px;border-top:1px solid var(--line)}
.slabel{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.4px;margin-bottom:10px}
.field{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
.field label{font-size:13px;color:var(--mut)}
.inp{background:var(--card);border:1px solid var(--line);border-radius:12px;color:var(--txt);font:inherit;padding:10px 12px;outline:none}
.inp:focus{border-color:var(--blue)}
.avedit{display:flex;align-items:center;gap:16px;margin-bottom:6px}
.avbig{width:60px;height:60px;border-radius:50%;background:var(--grad);background-size:cover;background-position:center;flex:0 0 auto}
.btn{background:var(--blue);color:#fff;border:0;border-radius:999px;padding:10px 18px;font:inherit;font-weight:600;cursor:pointer}
.btn:hover{filter:brightness(1.08)}.btn.ghost{background:transparent;border:1px solid var(--line);color:var(--txt)}
.rowline{display:flex;align-items:center;justify-content:space-between;padding:12px 0}
.sw{width:46px;height:26px;border-radius:999px;background:var(--line);position:relative;cursor:pointer;transition:background .2s;flex:0 0 auto}
.sw.on{background:var(--blue)}
.sw::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:transform .22s cubic-bezier(.3,1.4,.5,1)}
.sw.on::after{transform:translateX(20px)}
.dev{display:flex;align-items:center;gap:12px;padding:12px;border:1px solid var(--line);border-radius:16px;margin-bottom:10px;background:var(--card)}
.dev .dico{width:34px;height:34px;border-radius:11px;background:var(--hover);display:flex;align-items:center;justify-content:center;flex:0 0 auto}
.dev .dico svg{width:18px;height:18px;stroke:var(--mut);fill:none;stroke-width:2}
.dname{font-size:14px;color:var(--txt)}.dsub{font-size:12px;color:var(--dim)}
.dot2{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}
.dot2.on{background:#34a853}.dot2.off{background:#9aa0a6}
.bars{display:flex;gap:16px;margin-left:auto;text-align:right}
.bar{min-width:46px}.bar .bv{font-size:15px;color:var(--txt);font-weight:500}.bar .bl{font-size:11px;color:var(--dim)}
.prof{cursor:pointer}.prof:hover{background:var(--hover);border-radius:12px}
.steps{display:flex;gap:6px;margin:4px 0 14px}.pip{height:4px;flex:1;border-radius:2px;background:var(--line)}.pip.on{background:var(--blue)}
/* tabbed settings (no outer scroll) */
.modal.settings{width:min(720px,95vw);height:min(560px,90vh);padding:0;display:flex;overflow:hidden}
.snav{width:176px;background:var(--side);border-right:1px solid var(--line);padding:16px 10px;display:flex;flex-direction:column;gap:3px;flex:0 0 auto}
.snav .st{font-size:18px;font-weight:600;color:var(--txt);padding:2px 12px 14px}
.navitem{padding:10px 12px;border-radius:12px;cursor:pointer;color:var(--mut);font-size:14px;transition:background .15s}
.navitem:hover{background:var(--hover)}.navitem.on{background:var(--sel);color:var(--txt)}
.scontent{flex:1;padding:22px 26px;overflow:auto}.scontent .slabel:first-child{margin-top:0;padding-top:0;border:0}
select.inp{appearance:none;-webkit-appearance:none}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<nav id="side">
 <div class="shead"><div class="logo"><span class="spark">✦</span><span>Cortex</span></div></div>
 <div class="row" id="new" onclick="newChat()"><span class="ic"><svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></span><span>New chat</span></div>
 <div id="srch"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg><input id="srchin" placeholder="Search chats" oninput="renderChats()"></div>
 <div class="sect">Recent</div>
 <div id="list"></div>
 <div class="prof" onclick="openSettings()" title="Settings"><span class="pav" id="pav"></span><div class="pt"><div class="pname" id="pname">__OWNER__</div></div></div>
</nav>
<div id="main">
 <div id="top"><button class="mbtn" id="modelbtn" onclick="toggleMenu(event)">__MODEL__ ▾</button>
  <div id="menu"></div><span style="flex:1"></span>
  <div class="bellwrap"><button class="rbtn" id="bellbtn" title="Notifications" onclick="toggleNudges(event)"></button><span class="badge" id="nbadge"></span>
   <div id="nudgepanel"></div></div>
  <button class="rbtn" id="themebtn" title="Toggle light / dark" onclick="toggleThemeBtn(event)"></button></div>
 <div id="scroll"><div id="hello">Hello, __OWNER__</div><div id="wrap"></div></div>
 <div id="dock">
  <div id="attached"></div>
  <form id="form">
   <button type="button" class="rbtn" id="attach" title="Attach files (docs, images, video)" onclick="attachFiles()"><svg viewBox="0 0 24 24"><path d="M21 11.5 12.5 20a5 5 0 0 1-7-7l8.3-8.3a3.4 3.4 0 0 1 4.8 4.8L9.2 18.1a1.8 1.8 0 0 1-2.5-2.5l7.6-7.6"/></svg></button>
   <textarea id="q" rows="1" placeholder="Ask Cortex, or give it a job…"></textarea>
   <button type="button" class="rbtn" id="mic" title="Voice input (MLX)"><svg viewBox="0 0 24 24"><rect x="9" y="2.5" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0"/><path d="M12 17.5V21"/></svg></button>
   <button type="button" class="rbtn spk" id="spk" title="Read answers aloud (Kokoro-MLX)" onclick="toggleSpk()"><svg viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 6a9 9 0 0 1 0 12"/></svg></button>
   <button class="rbtn" id="send" title="Send"><svg viewBox="0 0 24 24"><path d="M12 20V5"/><path d="m5 12 7-7 7 7"/></svg></button>
  </form>
  <div class="foot">Cortex runs on-device and can make mistakes.</div>
 </div>
</div>
<div id="ov"><div class="modal" id="modal"></div></div>
<script>
const $=id=>document.getElementById(id);
const wrap=$('wrap'),form=$('form'),q=$('q'),send=$('send'),mic=$('mic'),hello=$('hello'),list=$('list');
let sid='gui-'+Date.now(),busy=false,speakOn=false,allChats=[],TABFILL='';
const PEN='<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
const DEFAULT_PH='Ask Cortex, or give it a job…';
function esc(s){return s.replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}
function el(t,c,h){const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e}
const CHEV='<svg class="chev" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>';
// typewriter — gives the live "thinking" a streaming feel instead of popping in chunks
function typeInto(e,text){e._k=(e._k||0)+1;const k=e._k;let i=0;const step=()=>{if(e._k!==k)return;e.textContent=text.slice(0,i);if(i<text.length){i+=2;setTimeout(step,11)}};step()}
function toggleSpk(){speakOn=!speakOn;$('spk').classList.toggle('on',speakOn);if(!speakOn&&window._a)window._a.pause()}
// real model selection (live switch of the Ollama model)
let curModel='';
async function loadModels(){try{const d=await (await fetch('/models')).json();curModel=d.current||'';
 const m=$('menu');m.innerHTML='';(d.models||[]).forEach(name=>{const it=el('div','mi',esc(name)+(name===curModel?'<span class="c">✓</span>':''));
  it.onclick=async()=>{m.classList.remove('show');$('modelbtn').textContent=shortModel(name)+' ▾';
   try{await fetch('/set_model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:name})});curModel=name;loadModels()}catch(_){}};
  m.appendChild(it)});
 if(curModel)$('modelbtn').textContent=shortModel(curModel)+' ▾';
}catch(_){}}
function shortModel(n){n=n.split('/').pop();if(n.includes('gemma4-agent')||n.includes('12b'))return 'Gemma 12B';return n.replace(':latest','')}
function toggleMenu(e){e.stopPropagation();const m=$('menu');m.classList.toggle('show');if(m.classList.contains('show'))loadModels()}
document.addEventListener('click',e=>{if(!e.target.closest('#menu')&&e.target.id!=='modelbtn')$('menu').classList.remove('show');
  if(!e.target.closest('.bellwrap'))$('nudgepanel').classList.remove('show')});
function newChat(){sid='gui-'+Date.now();wrap.innerHTML='';hello.style.display='';renderChats()}
q.addEventListener('input',()=>{q.style.height='auto';q.style.height=Math.min(q.scrollHeight,150)+'px';send.classList.toggle('show',q.value.trim().length>0)});
q.addEventListener('keydown',e=>{
 if(e.key==='Tab'&&TABFILL&&!q.value.trim()){e.preventDefault();q.value=TABFILL;q.dispatchEvent(new Event('input'));return}
 if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
// markdown
const SUP={'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','+':'⁺','-':'⁻','n':'ⁿ','i':'ⁱ','x':'ˣ','y':'ʸ','a':'ᵃ','b':'ᵇ'};
const SUB={'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉','+':'₊','-':'₋','n':'ₙ','x':'ₓ','i':'ᵢ','a':'ₐ'};
const SYM={pi:'π',theta:'θ',alpha:'α',beta:'β',gamma:'γ',delta:'δ',Delta:'Δ',lambda:'λ',mu:'μ',sigma:'σ',omega:'ω',phi:'φ',infty:'∞',times:'×',cdot:'·',div:'÷',pm:'±',leq:'≤',geq:'≥',le:'≤',ge:'≥',neq:'≠',approx:'≈',rightarrow:'→',Rightarrow:'⇒',to:'→',sum:'∑',int:'∫',partial:'∂',nabla:'∇',cdots:'⋯',ldots:'…'};
const sups=s=>[...s].map(c=>SUP[c]||c).join('');
const subz=s=>[...s].map(c=>SUB[c]||c).join('');
function tex(e){return e
 .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g,'($1)/($2)')
 .replace(/\\sqrt\s*\{([^{}]*)\}/g,'√($1)')
 .replace(/\^\{([^{}]*)\}/g,(_,p)=>sups(p)).replace(/\^(-?[0-9a-zA-Z])/g,(_,p)=>sups(p))
 .replace(/_\{([^{}]*)\}/g,(_,p)=>subz(p)).replace(/_(-?[0-9a-zA-Z])/g,(_,p)=>subz(p))
 .replace(/\\(pi|theta|alpha|beta|gamma|Delta|delta|lambda|mu|sigma|omega|phi|infty|times|cdot|div|pm|leq|geq|le|ge|neq|approx|rightarrow|Rightarrow|to|sum|int|partial|nabla|cdots|ldots)\b/g,(m,n)=>SYM[n]||m)
 .replace(/\\sqrt\b/g,'√').replace(/\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad/g,'')
 .replace(/\\\\/g,' ').replace(/[{}]/g,'').replace(/\\([a-zA-Z]+)/g,'$1').trim()}
function mathify(t){return t
 .replace(/\$\$([^$]+)\$\$/g,(_,e)=>tex(e))
 .replace(/\$([^$\n]+)\$/g,(_,e)=>tex(e))
 .replace(/\\\(([\s\S]*?)\\\)/g,(_,e)=>tex(e))
 .replace(/\\\[([\s\S]*?)\\\]/g,(_,e)=>tex(e))}
function md(src){src=src.replace(/\r/g,'');const out=[],L=src.split('\n');let i=0;
 const inl=t=>esc(mathify(t)).replace(/```/g,'').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/(^|[^*])\*([^*]+)\*/g,'$1<i>$2</i>').replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
 while(i<L.length){let ln=L[i];
  if(/^```/.test(ln)){let b=[];i++;while(i<L.length&&!/^```/.test(L[i])){b.push(esc(L[i]));i++}i++;out.push('<pre><code>'+b.join('\n')+'</code></pre>');continue}
  let h=ln.match(/^(#{1,3})\s+(.*)/);if(h){out.push('<h'+h[1].length+'>'+inl(h[2])+'</h'+h[1].length+'>');i++;continue}
  if(/^\s*[-*•]\s+/.test(ln)){let it=[];while(i<L.length&&/^\s*[-*•]\s+/.test(L[i])){it.push('<li>'+inl(L[i].replace(/^\s*[-*•]\s+/,''))+'</li>');i++}out.push('<ul>'+it.join('')+'</ul>');continue}
  if(/^\s*\d+\.\s+/.test(ln)){let it=[];while(i<L.length&&/^\s*\d+\.\s+/.test(L[i])){it.push('<li>'+inl(L[i].replace(/^\s*\d+\.\s+/,''))+'</li>');i++}out.push('<ol>'+it.join('')+'</ol>');continue}
  if(ln.trim()===''){i++;continue}
  let p=[ln];i++;while(i<L.length&&L[i].trim()!==''&&!/^(#{1,3}\s|```|\s*[-*•]\s|\s*\d+\.\s)/.test(L[i])){p.push(L[i]);i++}
  out.push('<p>'+inl(p.join(' '))+'</p>')}
 return out.join('')}
// sidebar chats (real history)
async function loadChats(){try{allChats=(await (await fetch('/chats')).json()).chats||[]}catch(_){allChats=[]}renderChats()}
function renderChats(){const f=($('srchin')||{}).value?$('srchin').value.toLowerCase():'';list.innerHTML='';
 allChats.filter(c=>!f||c.title.toLowerCase().includes(f)).forEach(c=>{
  const d=el('div','chat'+(c.session_id===sid?' sel':''),esc(c.title||'New chat'));
  d.onclick=()=>openChat(c.session_id);list.appendChild(d)})}
async function openChat(s){sid=s;hello.style.display='none';wrap.innerHTML='';renderChats();
 try{const msgs=(await (await fetch('/history?session='+encodeURIComponent(s))).json()).messages||[];
  msgs.forEach(m=>{if(m.role==='user')addUser(m.content);else addResp(m.content)})}catch(_){}
 $('scroll').scrollTop=$('scroll').scrollHeight}
function addUser(t){const turn=el('div','turn');const r=el('div','urow');
 const ed=el('span','uedit',PEN);ed.title='Edit & resend';ed.onclick=()=>editFrom(turn,t);r.appendChild(ed);
 const b=el('div','ubub');b.textContent=t;r.appendChild(b);turn.appendChild(r);wrap.appendChild(turn);return turn}
function editFrom(turn,text){if(busy)return;   // revert: drop this turn + everything after, refill input
 let n=turn;const rm=[];while(n){rm.push(n);n=n.nextElementSibling}rm.forEach(x=>x.remove());
 if(!wrap.children.length)hello.style.display='';
 q.value=text;q.dispatchEvent(new Event('input'));q.focus()}
function addResp(t){const d=el('div','resp ans');d.innerHTML=md(t);wrap.appendChild(d);return d}
// stream
function submit(text){if(busy||!text)return;busy=true;send.classList.remove('show');hello.style.display='none';
 TABFILL='';q.placeholder=DEFAULT_PH;
 const first=wrap.children.length===0;q.value='';q.style.height='auto';
 const turn=addUser(text);
 const resp=el('div','resp');
 const live=el('div','rlive active');
 const det=el('details','trace');          // collapsed by default → retractable; summary = live preview
 const sum=el('summary');sum.innerHTML='<span class="orb"></span><span class="jist shimmer"></span>'+CHEV;det.appendChild(sum);
 const body=el('div','body');det.appendChild(body);live.appendChild(det);
 const ans=el('div','ans');resp.appendChild(live);resp.appendChild(ans);turn.appendChild(resp);
 const sc=$('scroll');sc.scrollTop=sc.scrollHeight;
 const jist=sum.querySelector('.jist');
 const preview=t=>typeInto(jist,t);          // streaming-feel preview (the "jist")
 preview('Thinking…');
 const es=new EventSource('/stream?session='+encodeURIComponent(sid)+'&q='+encodeURIComponent(text));
 es.onmessage=e=>{let ev;try{ev=JSON.parse(e.data)}catch(_){return}
  if(ev.type==='status'){preview(ev.text)}
  else if(ev.type==='thinking'){preview(ev.text.replace(/\s+/g,' ').slice(0,110));const s=el('div','obs');s.textContent=ev.text;body.appendChild(s)}
  else if(ev.type==='tool'){preview('Using '+ev.name+(ev.detail?' · '+ev.detail:''));body.appendChild(el('div','step','<span class="k">'+esc(ev.name)+'</span><span class="d">'+esc(ev.detail||'')+'</span>'))}
  else if(ev.type==='observation'){const s=el('div','obs');s.textContent='→ '+ev.text;body.appendChild(s)}
  else if(ev.type==='image'){const card=el('div','chartcard');const im=el('img');im.src=ev.src;card.appendChild(im);if(ev.caption){const cp=el('div','chartcap');cp.textContent=ev.caption;card.appendChild(cp)}resp.appendChild(card)}
  else if(ev.type==='images'){const g=el('div','imgrid');(ev.items||[]).forEach(it=>{const a=el('a');a.href=it.page||it.url;a.target='_blank';const im=el('img');im.src=it.thumb||it.url;im.loading='lazy';im.alt=it.title||'';a.appendChild(im);g.appendChild(a)});resp.appendChild(g)}
  else if(ev.type==='answer'){live.classList.remove('active');jist.classList.remove('shimmer');jist.textContent='Show thinking';ans.innerHTML=md(ev.text);if(speakOn)ttsPlay(ev.text)}
  else if(ev.type==='ask'){const o=el('div','opts');(ev.options||[]).forEach(opt=>{const bt=el('div','opt',esc(opt));bt.onclick=()=>submit(opt);o.appendChild(bt)});resp.appendChild(o)}
  else if(ev.type==='followups'){const f=el('div','followups');(ev.items||[]).forEach((it,i)=>{const bt=el('div','fu');bt.innerHTML=esc(it)+(i===0?'<span class="kbd">Tab</span>':'');bt.onclick=()=>submit(it);f.appendChild(bt)});resp.appendChild(f);
   TABFILL=(ev.items&&ev.items[0])||'';if(TABFILL)q.placeholder='⇥ '+TABFILL}
  else if(ev.type==='done'){es.close();busy=false;if(first)loadChats();q.focus()}
  if(ev.type==='answer'||ev.type==='done'||det.open)sc.scrollTop=sc.scrollHeight};
 es.onerror=()=>{es.close();busy=false;if(!ans.innerHTML)ans.textContent='(connection dropped)'}}
form.addEventListener('submit',e=>{e.preventDefault();submit(q.value.trim())});
// voice: mic -> MLX whisper ; answers -> MLX kokoro
let rec=null,chunks=[],recording=false;
async function ttsPlay(text){try{const r=await fetch('/voice/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});if(!r.ok)return;const b=await r.blob();if(window._a)window._a.pause();window._a=new Audio(URL.createObjectURL(b));window._a.play()}catch(_){}}
mic.addEventListener('click',async()=>{if(recording){rec.stop();return}
 try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(stream);chunks=[];
  rec.ondataavailable=e=>chunks.push(e.data);
  rec.onstop=async()=>{mic.classList.remove('rec');recording=false;stream.getTracks().forEach(t=>t.stop());
   const fd=new FormData();fd.append('audio',new Blob(chunks,{type:'audio/webm'}),'speech.webm');
   try{const j=await (await fetch('/voice/stt',{method:'POST',body:fd})).json();if(j.text){submit(j.text)}}catch(_){}}
  rec.start();recording=true;mic.classList.add('rec')}catch(e){alert('Mic needs permission (Safari/Chrome).')}});
// ── theme: circular-reveal switch (View Transitions API), like the reference video ──
function applyTheme(t){document.documentElement.setAttribute('data-theme',t)}
applyTheme(localStorage.getItem('cx-theme')||'dark');
function setTheme(t,e){localStorage.setItem('cx-theme',t);
 fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({theme:t})}).catch(()=>{});
 const x=e?e.clientX:innerWidth-40,y=e?e.clientY:40,r=Math.hypot(Math.max(x,innerWidth-x),Math.max(y,innerHeight-y));
 if(!document.startViewTransition){applyTheme(t);return;}
 document.startViewTransition(()=>applyTheme(t)).ready.then(()=>{
  document.documentElement.animate({clipPath:[`circle(0px at ${x}px ${y}px)`,`circle(${r}px at ${x}px ${y}px)`]},
   {duration:520,easing:'ease-in-out',pseudoElement:'::view-transition-new(root)'})})}
// ── profile avatar / name ──
function refreshProfile(){fetch('/settings').then(r=>r.json()).then(s=>{
  const nm=s.name||'You';$('pname').textContent=nm;
  const h=$('hello');if(h)h.textContent='Hello, '+nm;   // keep the greeting in sync with the name
  if(s.has_avatar){$('pav').style.backgroundImage='url(/avatar?t='+Date.now()+')';$('pav').style.backgroundSize='cover';$('pav').style.backgroundPosition='center'}
  if(!localStorage.getItem('cx-theme')&&s.theme)applyTheme(s.theme);})}
// ── modal plumbing ──
const X='<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>';
function showModal(h,cls){const m=$('modal');m.className='modal'+(cls?' '+cls:'');m.innerHTML=h;$('ov').classList.add('show')}
function closeOv(){$('ov').classList.remove('show')}
$('ov').addEventListener('click',e=>{if(e.target.id==='ov')closeOv()});
const DICO={laptop:'<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M2 20h20"/></svg>',server:'<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="7" rx="1.5"/><rect x="3" y="13" width="18" height="7" rx="1.5"/><path d="M7 7.5h.01M7 16.5h.01"/></svg>'};
// ── settings ──
let CUR={};
async function openSettings(){CUR=await (await fetch('/settings')).json();
 showModal(`<button class="mx" onclick="closeOv()" title="Close">${X}</button>
   <div class="snav"><div class="st">Settings</div>
   <div class="navitem on" data-t="profile" onclick="sTab('profile')">Profile</div>
   <div class="navitem" data-t="devices" onclick="sTab('devices')">Devices</div>
   <div class="navitem" data-t="voice" onclick="sTab('voice')">Voice</div>
   <div class="navitem" data-t="providers" onclick="sTab('providers')">AI Providers</div>
   <div class="navitem" data-t="email" onclick="sTab('email')">Email</div>
   <div class="navitem" data-t="mcp" onclick="sTab('mcp')">MCP</div>
   <div class="navitem" data-t="jobs" onclick="sTab('jobs')">Jobs</div></div>
   <div class="scontent" id="scontent"></div>`,'settings');
 sTab('profile')}
function sTab(t){document.querySelectorAll('.navitem[data-t]').forEach(n=>n.classList.toggle('on',n.dataset.t===t));
 const c=$('scontent');
 if(t==='profile')renderProfile(c);
 else if(t==='devices')renderDevices(c);
 else if(t==='voice')renderVoice(c);
 else if(t==='providers')renderProviders(c);
 else if(t==='email')renderEmail(c);
 else if(t==='mcp')renderMcp(c);
 else renderJobs(c)}
const PROV=[['anthropic','Claude (Anthropic)','sk-ant-…','console.anthropic.com/settings/keys'],
            ['openai','OpenAI','sk-…','platform.openai.com/api-keys'],
            ['gemini','Gemini (Google)','AIza…','aistudio.google.com/apikey']];
function renderProviders(c){
 c.innerHTML=`<div class="slabel">AI providers</div>
  <div class="dsub" style="margin-bottom:12px;line-height:1.5">Cortex runs 100% locally by default. Paste an API key to also use a cloud model — keys are stored in your OS keychain, never in a file, and never leave this machine except to call that provider. After adding a key, pick its model from the selector top-left.</div>
  <div id="provlist"><div class="dsub">loading…</div></div>`;
 loadProviders()}
async function loadProviders(){const L=$('provlist');if(!L)return;
 let st={};try{st=(await (await fetch('/keys')).json()).providers||{}}catch(_){}
 L.innerHTML=PROV.map(([id,name,ph,url])=>{const on=st[id];
  return `<div class="dev"><div class="dico">${PLUG}</div><div style="flex:1;min-width:0">
    <div class="dname">${esc(name)} ${on?'<span class="dot2 on"></span>':''}</div>
    <div class="dsub">${on?'Connected ✓':'Not connected'} · <a href="https://${url}" target="_blank">get a key</a></div>
    <div style="display:flex;gap:8px;margin-top:8px">
     <input class="inp" id="k_${id}" placeholder="${ph}" autocomplete="off" style="flex:1">
     <button class="btn" onclick="saveKey('${id}')">${on?'Update':'Connect'}</button>
     ${on?`<span class="xbtn" title="Remove" onclick="rmKey('${id}')">${TRASH}</span>`:''}
    </div></div></div>`}).join('')+'<div class="dsub" id="pstat" style="margin-top:10px"></div>'}
async function saveKey(id){const k=$('k_'+id).value.trim();if(!k){$('pstat').textContent='Paste a key first.';return}
 $('pstat').textContent='Saving…';
 try{const r=await fetch('/keys/set',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:id,key:k})});
  if(!r.ok){$('pstat').textContent='Could not save that key.';return}
  $('pstat').textContent='Saved ✓ — pick its model from the selector top-left.';loadProviders();loadModels()}catch(_){$('pstat').textContent='Could not save that key.'}}
async function rmKey(id){await fetch('/keys/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:id})});loadProviders();loadModels()}
const PLUG='<svg viewBox="0 0 24 24"><path d="M9 2v6M15 2v6M7 8h10v3a5 5 0 0 1-10 0z"/><path d="M12 16v6"/></svg>';
function renderMcp(c){
 c.innerHTML=`<div class="slabel">MCP servers</div>
  <div class="dsub" style="margin-bottom:12px;line-height:1.5">Connect Cortex to any <b>MCP</b> server (same protocol Claude Code uses) — it discovers the server's tools and lets Cortex use them. Claude Code stays separate; configure that as usual.</div>
  <div id="mcplist"><div class="dsub">loading…</div></div>
  <div class="slabel">Add a server</div>
  <div class="field"><label>Name</label><input class="inp" id="mcpname" placeholder="filesystem" autocomplete="off"></div>
  <div class="field"><label>Command</label><input class="inp" id="mcpcmd" placeholder="npx -y @modelcontextprotocol/server-filesystem ~/Documents" autocomplete="off"></div>
  <div style="display:flex;gap:12px;align-items:center"><button class="btn" onclick="mcpAdd()">Add</button><span class="dsub" id="mcpstat"></span></div>`;
 loadMcp()}
async function loadMcp(){const L=$('mcplist');if(!L)return;
 try{const s=(await (await fetch('/mcp/servers')).json()).servers||[];
  L.innerHTML=s.length?s.map(x=>`<div class="dev"><div class="dico">${PLUG}</div><div style="min-width:0"><div class="dname">${esc(x.name)}</div><div class="dsub" id="mc${mid(x.name)}" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:320px">checking…</div></div><div class="bars"><span class="xbtn" onclick="mcpRemove('${esc(x.name)}')">${TRASH}</span></div></div>`).join(''):'<div class="dsub">No MCP servers connected yet.</div>';
  if(s.length){const st=(await (await fetch('/mcp/status')).json()).status||[];st.forEach(x=>{const e=$('mc'+mid(x.name));if(e)e.textContent=x.connected?(x.tools+' tools connected'):'not connected — check the command'})}
 }catch(_){L.innerHTML='<div class="dsub">could not load</div>'}}
async function mcpAdd(){const name=$('mcpname').value.trim(),cmd=$('mcpcmd').value.trim();
 if(!name||!cmd){$('mcpstat').textContent='Enter a name and command.';return}
 $('mcpstat').textContent='Adding & connecting…';
 try{const j=await (await fetch('/mcp/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,command:cmd})})).json();
  if(j.ok){$('mcpname').value='';$('mcpcmd').value='';$('mcpstat').textContent='Added ✓';loadMcp()}else{$('mcpstat').textContent=j.error||'failed'}}catch(_){$('mcpstat').textContent='error'}}
async function mcpRemove(name){await fetch('/mcp/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});loadMcp()}
const MAILIC='<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>';
const TRASH='<svg viewBox="0 0 24 24"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>';
function mid(e){return e.replace(/[^a-z0-9]/gi,'')}
function renderEmail(c){
 c.innerHTML=`<div class="slabel">Gmail accounts</div>
  <div class="dsub" style="margin-bottom:12px;line-height:1.5">Connect with a Gmail <b>app password</b> (needs 2-Step Verification). Create one at <a href="https://myaccount.google.com/apppasswords" target="_blank">myaccount.google.com/apppasswords</a> and paste the 16-character code. It's stored in your Keychain — never in a file.</div>
  <div id="maillist"><div class="dsub">loading…</div></div>
  <div class="slabel">Add a Gmail account</div>
  <div class="field"><label>Gmail address</label><input class="inp" id="maddr" placeholder="you@gmail.com" autocomplete="off"></div>
  <div class="field"><label>App password</label><input class="inp" id="mpw" placeholder="abcd efgh ijkl mnop" autocomplete="off"></div>
  <div style="display:flex;gap:12px;align-items:center"><button class="btn" onclick="mailAdd()">Connect</button><span class="dsub" id="mstat"></span></div>`;
 loadMail()}
async function loadMail(){const L=$('maillist');if(!L)return;
 try{const a=(await (await fetch('/mail/accounts')).json()).accounts||[];
  L.innerHTML=a.length?a.map(x=>`<div class="dev"><div class="dico">${MAILIC}</div><div><div class="dname">${esc(x.email)}</div><div class="dsub" id="u${mid(x.email)}">checking…</div></div><div class="bars"><span class="xbtn" title="Remove" onclick="mailRemove('${esc(x.email)}')">${TRASH}</span></div></div>`).join(''):'<div class="dsub">No accounts connected yet.</div>';
  if(a.length){const u=(await (await fetch('/mail/unread')).json()).accounts||[];u.forEach(x=>{const e=$('u'+mid(x.email));if(e)e.textContent=x.unread>=0?x.unread+' unread':'could not reach (check app password)'})}
 }catch(_){L.innerHTML='<div class="dsub">could not load accounts</div>'}}
async function mailAdd(){const addr=$('maddr').value.trim(),pw=$('mpw').value.trim();
 if(!addr||!pw){$('mstat').textContent='Enter the address and app password.';return}
 $('mstat').textContent='Connecting…';
 try{const j=await (await fetch('/mail/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:addr,password:pw})})).json();
  if(j.ok){$('maddr').value='';$('mpw').value='';$('mstat').textContent='Connected ✓';loadMail()}else{$('mstat').textContent=j.error||'Could not connect'}}catch(_){$('mstat').textContent='Connection error'}}
async function mailRemove(addr){await fetch('/mail/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:addr})});loadMail()}
function renderProfile(c){
 c.innerHTML=`<div class="slabel">Profile</div>
  <div class="avedit"><span class="avbig" id="savbig"></span><button class="btn ghost" onclick="cxAv('savbig')">Change photo</button></div>
  <div class="field"><label>Your name</label><input class="inp" id="setname" value="${esc(CUR.name||'')}"></div>
  <div style="display:flex;justify-content:flex-end"><button class="btn" onclick="saveProfile()">Save</button></div>`;
 if(CUR.has_avatar)setBg('savbig')}
function renderDevices(c){const sv=CUR.server||{};
 c.innerHTML=`<div class="slabel">Connected devices &amp; status</div><div id="devlist"><div class="dsub">loading…</div></div>
  <div class="slabel">Server</div>
  <div class="field"><label>Label</label><input class="inp" id="srvlabel" value="${esc(sv.label||'')}" placeholder="myserver"></div>
  <div class="field"><label>Host / Tailscale IP</label><input class="inp" id="srvhost" value="${esc(sv.host||'')}" placeholder="100.x.x.x"></div>
  <div class="field"><label>SSH user</label><input class="inp" id="srvuser" value="${esc(sv.user||'')}"></div>
  <div style="display:flex;justify-content:flex-end"><button class="btn" onclick="saveServer()">Save</button></div>`;
 loadDevices()}
const VOICES=[['bm_daniel','Daniel · British male'],['bm_george','George · British male'],['bf_emma','Emma · British female'],['bf_isabella','Isabella · British female'],['am_adam','Adam · American male'],['am_michael','Michael · American male'],['af_heart','Heart · American female'],['af_bella','Bella · American female'],['af_nicole','Nicole · American female'],['af_sarah','Sarah · American female']];
function renderVoice(c){const cv=CUR.voice||'bm_daniel';
 c.innerHTML=`<div class="slabel">Cortex's voice · Kokoro (MLX)</div>
  <div class="field"><label>Voice</label><select class="inp" id="voicesel" onchange="saveVoice()">${VOICES.map(v=>`<option value="${v[0]}" ${v[0]===cv?'selected':''}>${v[1]}</option>`).join('')}</select></div>
  <div style="display:flex;gap:10px;margin-top:4px"><button class="btn ghost" onclick="testVoice()">▶ Test voice</button></div>
  <div class="dsub" id="vstat" style="margin-top:14px">Used for spoken replies (🔊) and voice mode.</div>`}
async function saveVoice(){CUR.voice=$('voicesel').value;await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({voice:CUR.voice})})}
async function testVoice(){const v=$('voicesel').value;$('vstat').textContent='Synthesizing…';
 try{const r=await fetch('/voice/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:"Hey, this is how I sound.",voice:v})});
  if(!r.ok){$('vstat').textContent='Voice server busy — try again in a moment.';return}
  const b=await r.blob();if(window._a)window._a.pause();window._a=new Audio(URL.createObjectURL(b));window._a.play();$('vstat').textContent='Playing a sample…'}catch(_){$('vstat').textContent='Could not play the sample.'}}
const CLOCK='<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>';
function renderJobs(c){c.innerHTML=`<div class="slabel">Running jobs</div><div id="joblist"><div class="dsub">loading…</div></div>
  <div class="slabel">Add a recurring task</div>
  <div class="dsub" style="margin-bottom:10px;line-height:1.5">Cortex re-runs the instruction on a schedule and delivers the result. e.g. <i>"summarize my unread email"</i> · <i>daily 08:00</i>.</div>
  <div class="field"><label>Instruction</label><input class="inp" id="tprompt" placeholder="what should Cortex do each time?" autocomplete="off"></div>
  <div style="display:flex;gap:10px"><div class="field" style="flex:1"><label>Schedule</label><input class="inp" id="tsched" placeholder="daily 08:00 · every 30 min · hourly" autocomplete="off"></div>
   <div class="field" style="width:130px"><label>Deliver via</label><select class="inp" id="tdeliver"><option value="notify">Notification</option><option value="email">Email</option><option value="both">Both</option></select></div></div>
  <div style="display:flex;gap:12px;align-items:center"><button class="btn" onclick="addTask()">Schedule</button><span class="dsub" id="tstat"></span></div>`;loadJobs()}
async function loadJobs(){try{const d=await (await fetch('/jobs')).json();
  let h=(d.scheduled||[]).map(j=>`<div class="dev"><div class="dico">${CLOCK}</div><div><div class="dname">${esc(j.name)}</div><div class="dsub"><span class="dot2 ${j.running?'on':'off'}"></span>${esc(j.desc)}</div></div><div class="bars"><div class="bar"><div class="bv" style="font-size:12px;color:${j.running?'#34a853':'var(--dim)'}">${j.running?'running':'idle'}</div></div></div></div>`).join('');
  if((d.tasks||[]).length){h+='<div class="slabel">Your recurring tasks</div>'+d.tasks.map(t=>`<div class="dev"><div class="dico">${CLOCK}</div><div style="min-width:0"><div class="dname">${esc(t.desc)}</div><div class="dsub" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:300px">${esc(t.human||t.schedule)} · ${esc(t.deliver||'notify')}</div></div><div class="bars"><span class="xbtn" title="Remove" onclick="removeTask('${esc(t.id)}')">${TRASH}</span></div></div>`).join('')}
  if((d.reminders||[]).length){h+='<div class="slabel">Scheduled reminders</div>'+d.reminders.map(r=>`<div class="dev"><div class="dico">${CLOCK}</div><div><div class="dname">${esc(r.text)}</div><div class="dsub">${esc(r.due)}</div></div></div>`).join('')}
  if((d.goals||[]).length){h+='<div class="slabel">Active goals</div>'+d.goals.map(g=>`<div class="dev" style="padding:11px 12px"><div class="dname" style="font-size:13px;line-height:1.45">• ${esc(g)}</div></div>`).join('')}
  $('joblist').innerHTML=h||'<div class="dsub">nothing scheduled</div>'}catch(_){$('joblist').innerHTML='<div class="dsub">could not load jobs</div>'}}
async function addTask(){const p=$('tprompt').value.trim(),s=$('tsched').value.trim();
  if(!p||!s){$('tstat').textContent='Need an instruction and a schedule.';return}
  $('tstat').textContent='Scheduling…';
  try{const r=await fetch('/tasks/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({desc:p.slice(0,50),prompt:p,schedule:s,deliver:$('tdeliver').value})});
   if(!r.ok){$('tstat').textContent='Could not schedule that.';return}
   $('tprompt').value='';$('tsched').value='';$('tstat').textContent='Scheduled.';loadJobs()}catch(_){$('tstat').textContent='Could not schedule that.'}}
async function removeTask(id){await fetch('/tasks/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});loadJobs()}
function setBg(id){const e=$(id);if(e){e.style.backgroundImage='url(/avatar?t='+Date.now()+')';e.style.backgroundSize='cover';e.style.backgroundPosition='center'}}
// theme toggle lives OUTSIDE settings (top bar)
const MOON='<svg viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
const SUN='<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
function updateThemeIcon(){const t=document.documentElement.getAttribute('data-theme')||'dark';const b=$('themebtn');if(b)b.innerHTML=t==='dark'?MOON:SUN}
function toggleThemeBtn(e){const cur=document.documentElement.getAttribute('data-theme')||'dark';setTheme(cur==='dark'?'light':'dark',e);setTimeout(updateThemeIcon,40)}
// ── notification bell (proactive nudges) ──
const BELL='<svg viewBox="0 0 24 24"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>';
let NUDGES=[];
function initBell(){const b=$('bellbtn');if(b)b.innerHTML=BELL;loadNudges();setInterval(loadNudges,60000)}
async function loadNudges(){try{const d=await (await fetch('/nudges')).json();NUDGES=d.nudges||[];
  const bg=$('nbadge');if(bg){if(d.unread>0){bg.textContent=d.unread>9?'9+':d.unread;bg.classList.add('show')}else bg.classList.remove('show')}
  if($('nudgepanel').classList.contains('show'))renderNudges()}catch(_){}}
function renderNudges(){const p=$('nudgepanel');
  const items=NUDGES.length?NUDGES.map(n=>`<div class="nitem ${n.read?'':'un'}"><div class="nt"><span class="ndot ${esc(n.kind)}"></span>${esc(n.title)}</div>${n.body?`<div class="nb">${esc(n.body)}</div>`:''}<div class="nw">${esc(n.when).slice(0,16).replace('T',' ')}</div></div>`).join(''):'<div class="nitem"><div class="nb">Nothing new — Cortex will nudge you here when it notices something.</div></div>';
  p.innerHTML=`<div class="nhead"><b>Notifications</b>${NUDGES.length?'<span onclick="clearNudges()">Clear all</span>':''}</div>${items}`}
function toggleNudges(e){e.stopPropagation();const p=$('nudgepanel');const open=p.classList.toggle('show');
  if(open){renderNudges();fetch('/nudges/read',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(()=>{$('nbadge').classList.remove('show')})}}
async function clearNudges(){await fetch('/nudges/clear',{method:'POST'});NUDGES=[];renderNudges();$('nbadge').classList.remove('show')}
function cxAv(targetId){const i=document.createElement('input');i.type='file';i.accept='image/*';
 i.onchange=async()=>{const f=i.files[0];if(!f)return;const fd=new FormData();fd.append('avatar',f);await fetch('/avatar',{method:'POST',body:fd});setBg(targetId);refreshProfile()};i.click()}
async function saveProfile(){CUR.name=$('setname').value.trim();
 await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:CUR.name})});refreshProfile()}
async function saveServer(){CUR.server={label:$('srvlabel').value.trim(),host:$('srvhost').value.trim(),user:$('srvuser').value.trim()};
 await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({server:CUR.server})});loadDevices()}
function devCard(x){const on=x.status==='online';
 const bars=[['cpu','CPU'],['mem','RAM']].filter(k=>x[k[0]]!=null).map(k=>`<div class="bar"><div class="bv">${x[k[0]]}%</div><div class="bl">${k[1]}</div></div>`).join('');
 return `<div class="dev"><div class="dico">${DICO[x.kind]||DICO.laptop}</div>
  <div><div class="dname">${esc(x.label)}</div><div class="dsub"><span class="dot2 ${on?'on':'off'}"></span>${esc(x.detail||(on?'online':'offline'))}</div></div>
  <div class="bars">${bars}</div></div>`}
async function loadDevices(){const L=$('devlist');if(!L)return;
 try{const d=(await (await fetch('/devices')).json()).devices||[];
  L.innerHTML=d.map(devCard).join('')+'<div class="dev" id="srvrow"><div class="dico">'+DICO.server+'</div><div><div class="dname">checking server…</div><div class="dsub"><span class="spin"></span>reaching it over Tailscale</div></div></div>';
 }catch(_){L.innerHTML='<div class="dsub">could not load devices</div>'}
 try{const s=(await (await fetch('/device_server')).json()).device;const row=$('srvrow');if(row)row.outerHTML=devCard(s)}catch(_){const row=$('srvrow');if(row)row.remove()}}
// ── onboarding (first run) ──
let OB={};
async function maybeOnboard(){const s=await (await fetch('/settings')).json();if(s.onboarded)return;OB={name:s.name||'',server:s.server||{},has_avatar:s.has_avatar};obStep(0)}
function pips(n){return `<div class="steps"><div class="pip ${n>=0?'on':''}"></div><div class="pip ${n>=1?'on':''}"></div><div class="pip ${n>=2?'on':''}"></div></div>`}
function obStep(n){
 if(n===0){showModal(`<h2>Hey — I'm Cortex ✦</h2><div class="sub">Your brother in the machine. Quick setup — about 20 seconds.</div>${pips(0)}
   <div class="field"><label>What should I call you?</label><input class="inp" id="obname" value="${esc(OB.name)}" placeholder="Your name"></div>
   <div style="display:flex;justify-content:flex-end;margin-top:14px"><button class="btn" onclick="OB.name=document.getElementById('obname').value.trim();obStep(1)">Next</button></div>`);
  setTimeout(()=>{const e=$('obname');if(e)e.focus()},60)}
 else if(n===1){showModal(`<h2>Add a photo</h2><div class="sub">Optional — makes it yours.</div>${pips(1)}
   <div class="avedit"><span class="avbig" id="obav"></span><button class="btn ghost" onclick="obAv()">Choose photo</button></div>
   <div style="display:flex;justify-content:space-between;margin-top:14px"><button class="btn ghost" onclick="obStep(0)">Back</button><button class="btn" onclick="obStep(2)">Next</button></div>`);
  if(OB.has_avatar)setBg('obav')}
 else{const sv=OB.server||{};showModal(`<h2>Your server &amp; devices</h2><div class="sub">So I can keep an eye on your infrastructure. Skip and add later anytime.</div>${pips(2)}
   <div class="field"><label>Server label</label><input class="inp" id="obslabel" value="${esc(sv.label||'')}" placeholder="myserver"></div>
   <div class="field"><label>Host / Tailscale IP</label><input class="inp" id="obshost" value="${esc(sv.host||'')}" placeholder="100.x.x.x"></div>
   <div class="field"><label>SSH user</label><input class="inp" id="obsuser" value="${esc(sv.user||'')}"></div>
   <div style="display:flex;justify-content:space-between;margin-top:14px"><button class="btn ghost" onclick="obStep(1)">Back</button><button class="btn" onclick="obFinish()">Finish</button></div>`)}}
function obAv(){const i=document.createElement('input');i.type='file';i.accept='image/*';
 i.onchange=async()=>{const f=i.files[0];if(!f)return;const fd=new FormData();fd.append('avatar',f);await fetch('/avatar',{method:'POST',body:fd});OB.has_avatar=true;setBg('obav')};i.click()}
async function obFinish(){OB.server={label:$('obslabel').value.trim(),host:$('obshost').value.trim(),user:$('obsuser').value.trim()};
 await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:OB.name||'',server:OB.server,onboarded:true})});
 closeOv();refreshProfile()}
function attachFiles(){const i=document.createElement('input');i.type='file';i.multiple=true;
 i.onchange=async()=>{if(!i.files.length)return;const fd=new FormData();for(const f of i.files)fd.append('files',f);
  const tag=el('div','fu');tag.innerHTML='<span class="spin"></span>reading '+i.files.length+' file(s)…';$('attached').appendChild(tag);
  try{const j=await (await fetch('/files/upload',{method:'POST',body:fd})).json();tag.remove();
   (j.files||[]).forEach(x=>{const c=el('div','fu');c.textContent='📎 '+x.name+(x.chunks?' · '+x.chunks+' chunks':' · '+(x.error||'?'));$('attached').appendChild(c)})}catch(_){tag.textContent='upload failed'}};
 i.click()}
// Unregister any previously-installed service worker + clear its caches (it used
// to pin stale app code). We deliberately do NOT register a new one — Cortex needs
// the local server running anyway, so there's nothing to gain from shell caching.
if('serviceWorker' in navigator){
  navigator.serviceWorker.getRegistrations().then(rs=>rs.forEach(r=>r.unregister())).catch(()=>{});
  if(window.caches)caches.keys().then(ks=>ks.forEach(k=>caches.delete(k))).catch(()=>{});
}
loadChats();loadModels();refreshProfile();updateThemeIcon();initBell();maybeOnboard();
</script></body></html>"""


def _open_browser(url):
    import cortex.util.osutil as osutil
    osutil.open_url(url)


def _port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    import threading
    port = int(os.environ.get("CORTEX_GUI_PORT", "7860"))
    url = f"http://127.0.0.1:{port}"
    # Only auto-open the browser when explicitly asked (the installer sets this on
    # first run). `cx gui` does NOT open a tab — it just serves and prints the URL.
    auto_open = os.environ.get("CORTEX_GUI_OPEN", "").lower() in ("1", "true", "yes")
    if _port_in_use(port):
        logger.info(f"Cortex GUI already running at {url}")
        if auto_open:
            _open_browser(url)
        return
    logger.info(f"Cortex GUI running at {url}  (open it in your browser; Ctrl+C to stop)")
    if auto_open:
        threading.Timer(1.0, lambda: _open_browser(url)).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
