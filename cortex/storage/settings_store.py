"""
settings_store.py — local user settings + profile picture for the Cortex GUI.

Everything lives under ~/.cortex (same place as the DB and secrets), stays on
device, and is shared with the rest of Cortex (owner name feeds the persona).
"""
from __future__ import annotations

import os
import json

_DIR = os.path.expanduser("~/.cortex")
SETTINGS_FILE = os.path.join(_DIR, "settings.json")
OWNER_FILE = os.path.join(_DIR, "owner.json")
ONBOARD_FLAG = os.path.join(_DIR, ".onboarded")
AVATAR_FILE = os.path.join(_DIR, "avatar.png")

_DEFAULTS = {
    "name": "",
    "theme": "dark",            # dark | light
    "voice": "bm_daniel",       # Kokoro-MLX TTS voice
    "onboarded": False,
    "server": {"host": "", "user": "", "label": ""},
    "devices": [],              # extra devices the user added: [{label,host,user}]
    "emails": [],               # connected Gmail accounts: [{email,label}] (app pw in Keychain)
    "mcp_servers": [],          # MCP servers: [{name, command, env}]
    "tasks": [],                # user-scheduled recurring tasks: [{id,desc,prompt,schedule,deliver,enabled}]
    # Subagents: Cortex splitting a big job into focused helpers. mode = auto (decided per
    # task) | off | eager (consider splitting every non-trivial request). max_parallel 0 = auto (1 for a local model, up to 3 for cloud).
    # approvals = how much runs without asking: careful | balanced (see cortex/core/safety.py).
    "agents": {"subagents": "auto", "max_parallel": 0, "max_helpers": 4, "helper_steps": 4,
               "approvals": "careful"},
    "llm": {"model": ""},       # the chosen model; empty = config.yaml's default
    # Cloud spending limits; 0 = no limit. Local models are never limited. A model with no
    # known price is held by the token limits. See cortex/core/usage.py.
    "limits": {"daily_usd": 5.0, "monthly_usd": 30.0, "job_usd": 1.0,
               "daily_tokens": 2000000, "job_tokens": 500000, "prices": {}},
    # Resource controls. keep_alive = how long the AI model stays in memory after a reply
    # ("0" = free it right away, "-1" = never unload). followups = auto|off|always.
    "resources": {"keep_alive": "2m", "followups": "auto", "ai_jobs_only_on_ac": True,
                  "voice_idle_minutes": 5},
}

# Nested sections whose missing keys fall back to the defaults above.
_SECTIONS = ("agents", "resources", "llm", "limits")


def load() -> dict:
    s = dict(_DEFAULTS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            s.update(json.load(f))
    except (FileNotFoundError, ValueError):
        pass
    # name falls back to owner.json (set by the CLI onboarding)
    if not s.get("name"):
        try:
            with open(OWNER_FILE, encoding="utf-8") as f:
                s["name"] = json.load(f).get("name", "") or ""
        except Exception:
            pass
    for sec in _SECTIONS:   # a partial section in the file still gets every default key
        s[sec] = {**_DEFAULTS[sec], **(s.get(sec) or {})}
    s["has_avatar"] = os.path.exists(AVATAR_FILE)
    return s


def section(name: str) -> dict:
    """The agents / resources settings, with defaults filled in."""
    return load().get(name, {})


def update_section(name: str, patch: dict) -> dict:
    """Merge `patch` into one nested section and persist it."""
    cur = dict(section(name))
    cur.update({k: v for k, v in (patch or {}).items() if k in _DEFAULTS[name]})
    save({name: cur})
    return cur


def save(patch: dict) -> dict:
    os.makedirs(_DIR, mode=0o700, exist_ok=True)
    s = load()
    for k, v in (patch or {}).items():
        if k in ("name", "theme", "voice", "onboarded", "server", "devices",
                 "emails", "mcp_servers", "tasks", "agents", "resources", "llm", "limits"):
            s[k] = v
    s.pop("has_avatar", None)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    # keep owner.json (persona) + the CLI onboarding flag in sync
    if s.get("name"):
        with open(OWNER_FILE, "w", encoding="utf-8") as f:
            json.dump({"name": s["name"]}, f)
    if s.get("onboarded"):
        open(ONBOARD_FLAG, "w", encoding="utf-8").close()
    return load()


def save_avatar(raw: bytes) -> None:
    os.makedirs(_DIR, mode=0o700, exist_ok=True)
    with open(AVATAR_FILE, "wb") as f:
        f.write(raw)
