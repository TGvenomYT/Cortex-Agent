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
    "tasks": [],                # user-scheduled recurring tasks: [{id,desc,prompt,schedule,deliver}]
}


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
    s["has_avatar"] = os.path.exists(AVATAR_FILE)
    return s


def save(patch: dict) -> dict:
    os.makedirs(_DIR, mode=0o700, exist_ok=True)
    s = load()
    for k, v in (patch or {}).items():
        if k in ("name", "theme", "voice", "onboarded", "server", "devices",
                 "emails", "mcp_servers", "tasks"):
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
