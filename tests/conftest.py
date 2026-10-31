"""Every test runs against a throwaway home folder and never touches launchd or the Keychain."""
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Stands in for launchctl, so no test can load a real job on this Mac.
FAKE_LAUNCHCTL = """#!/bin/sh
d="$HOME/.fake-launchd"; mkdir -p "$d"
case "$1" in
  load) [ -e "$HOME/.launchd-refuses" ] || touch "$d/$(basename "$2" .plist)" ;;
  unload) rm -f "$d/$(basename "$2" .plist)" ;;
  list) printf 'PID\\tStatus\\tLabel\\n'
        for f in "$d"/*; do [ -e "$f" ] && printf -- '-\\t0\\t%s\\n' "$(basename "$f")"; done ;;
esac
exit 0
"""


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    import cortex.storage.settings_store as ss
    import cortex.util.osutil as osutil
    from cortex.core import scheduler
    monkeypatch.setenv("HOME", str(tmp_path))
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    fake = bin_dir / "launchctl"
    fake.write_text(FAKE_LAUNCHCTL)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(scheduler, "detach", lambda argv, name: None)
    monkeypatch.setattr(ss, "_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(ss, "OWNER_FILE", str(tmp_path / "owner.json"))
    monkeypatch.setattr(ss, "ONBOARD_FLAG", str(tmp_path / ".onboarded"))
    monkeypatch.setattr(ss, "AVATAR_FILE", str(tmp_path / "avatar.png"))
    secrets = {}
    monkeypatch.setattr(osutil, "secret_get", lambda a: secrets.get(a))
    monkeypatch.setattr(osutil, "secret_set", lambda a, v: secrets.__setitem__(a, v) or True)
    monkeypatch.setattr(osutil, "secret_delete", lambda a: secrets.pop(a, None))
    import cortex.llm.embeddings as emb
    monkeypatch.setattr(emb, "_embed_uncached", lambda text: None)
    import cortex.llm.llm as llm
    llm._MODEL_CACHE.clear()
    return tmp_path


class Scripted:
    """A model that says what the test tells it to, and records what it was shown."""
    primary = None

    def __init__(self, *turns):
        self.turns, self.seen = list(turns), []

    def complete(self, messages, json_mode=False, schema=None):
        return "{}"

    def complete_with_tools(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        turn = self.turns.pop(0) if self.turns else "all done"
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, str):
            return {"content": turn, "tool_calls": [], "thinking": ""}
        return {"content": turn.get("say", ""), "thinking": "",
                "tool_calls": [{"id": f"c{i}", "function": {"name": n, "arguments": a}}
                               for i, (n, a) in enumerate(turn["calls"])]}


def shell(cmd):
    return {"calls": [("execute_shell", {"command": cmd})]}


@pytest.fixture
def make_core(tmp_path, monkeypatch):
    from cortex.core import cortex_core

    def make(*turns):
        config = {
            "database": {"backend": "sqlite", "sqlite_path": str(tmp_path / "cortex.db")},
            "llm": {"ollama": {"model": "fake:1b", "base_url": "http://127.0.0.1:9"}},
            "context": {"simple_mode": False}, "memory": {"summarize_every": 2}, "autonomy": {},
        }
        monkeypatch.setattr("cortex.util.cortex_bootstrap.load_secrets", lambda: None)
        core = cortex_core.CortexCore(config)
        core.mcp = None
        core.llm = Scripted(*turns)
        core.subagents.assess = lambda text: None
        monkeypatch.setattr(core, "_want_followups", lambda answer: False)
        return core
    return make
