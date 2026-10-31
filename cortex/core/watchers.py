"""
watchers.py — look after something that's running, and come back without being asked.

Each watch is its own small process, separate from the app, so it keeps going after a chat
ends or the window closes. It uses no AI while it waits. It polls something cheap, and
when something happens it wakes the model once, writes the result into the chat the
request came from, and raises a notification.

What can be watched:
    pid       a process by id (including a command Cortex moved to the background)
    process   a process by name
    port      a port opening or closing
    file      a file appearing, or text appearing in a log
    check     a read-only command, until it succeeds, fails, changes or shows some text

Run directly as:  python -m cortex.core.watchers <id>
"""
from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import time
import uuid

logger = logging.getLogger("cortex.watchers")

STALL_MINUTES = 15      # a log silent this long gets one "may be stuck" message
MAX_HOURS = 24
INTERVAL = {"pid": 5, "process": 10, "port": 10, "file": 10, "check": 30}


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":     # os.kill(pid, 0) there would send a Ctrl+C
        from cortex.core.runtime import _alive
        return _alive(pid)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    try:    # a finished child that nobody has collected yet still answers
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def _matching(pattern: str) -> list:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if pattern.lower() in line.lower()]
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []
    return [p for p in out if int(p) != os.getpid()]


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex((host, int(port))) == 0


def _tail(path: str, lines: int = 30) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 8000))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""


class Probe:
    """Looks at the target once per tick. step() -> an event dict, or None."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.kind = spec["kind"]
        self.log = spec.get("log")
        self.offset = 0
        self.last_output = None
        self.log_size = -1
        self.log_changed = time.time()
        self.warned_stall = False

    def problem(self) -> str | None:
        """Why this can't be watched as asked, checked once up front."""
        s = self.spec
        if self.kind == "pid" and not s.get("exit_file") and not _pid_alive(s["pid"]):
            return f"there's no running process {s['pid']}"
        if self.kind == "process" and not _matching(s["pattern"]):
            return f"nothing running matches '{s['pattern']}'"
        return None

    def output(self) -> str:
        """The latest output, errors included: that's usually where the reason is."""
        parts = [_tail(p) for p in (self.spec.get("log"), self.spec.get("err_log")) if p]
        return "\n".join(x for x in parts if x)

    def _finished(self, detail: str, code=None) -> dict:
        return {"what": "finished", "final": True, "detail": detail, "code": code, "tail": self.output()}

    def step(self) -> dict | None:
        event = getattr(self, f"_{self.kind}")()
        return event or self._stalled()

    def _pid(self):
        s = self.spec
        path = s.get("exit_file")
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    code = int(f.read().strip() or 0)
            except (OSError, ValueError):
                code = None
            word = "finished successfully" if code == 0 else f"failed with exit code {code}"
            return self._finished(word, code)
        if not _pid_alive(s["pid"]):
            return self._finished("has stopped, but there's no way to tell from here whether it succeeded")
        return None

    def _process(self):
        if _matching(self.spec["pattern"]):
            return None
        return self._finished("is no longer running, but there's no way to tell from here whether it succeeded")

    def _port(self):
        want_open = self.spec.get("until") == "open"
        if _port_open(self.spec["port"]) == want_open:
            return self._finished(f"port {self.spec['port']} is {'open' if want_open else 'closed'}")
        return None

    def _file(self):
        s, path = self.spec, self.spec["path"]
        if not os.path.exists(path):
            return None
        pattern = s.get("pattern")
        if not pattern:
            return self._finished("now exists")
        with open(path, "rb") as f:
            f.seek(self.offset)
            text = f.read().decode("utf-8", "replace")
            self.offset = f.tell()
        m = re.search(pattern, text, re.I)
        if not m:
            return None
        line = text[text.rfind("\n", 0, m.start()) + 1:].split("\n")[0]
        return {"what": "text", "final": True, "detail": f"the log shows: {line[:200]}", "tail": _tail(path)}

    def _check(self):
        s = self.spec
        from cortex.llm.viz import clean_env
        try:
            r = subprocess.run(s["command"], shell=True, capture_output=True, text=True, timeout=60, env=clean_env())
        except subprocess.TimeoutExpired:
            return None
        out = (r.stdout or "") + (r.stderr or "")
        until = s.get("until") or "changes"
        hit = False
        if until == "success":
            hit = r.returncode == 0
        elif until == "fail":
            hit = r.returncode != 0
        elif until.startswith("contains:"):
            hit = until[9:].lower() in out.lower()
        elif until.startswith("gone:"):
            hit = until[5:].lower() not in out.lower()
        else:
            hit = self.last_output is not None and out != self.last_output
        self.last_output = out
        if hit:
            return {"what": "changed", "final": True, "detail": f"the check says {until}", "tail": out[-1500:]}
        return None

    def _stalled(self):
        if not self.log or not os.path.exists(self.log):
            return None
        size = os.path.getsize(self.log)
        if size != self.log_size:
            self.log_size, self.log_changed, self.warned_stall = size, time.time(), False
            return None
        minutes = self.spec.get("stall_minutes", STALL_MINUTES)
        if not self.warned_stall and minutes and time.time() - self.log_changed > minutes * 60:
            self.warned_stall = True
            return {"what": "stalled", "final": False, "tail": self.output(),
                    "detail": f"has written nothing for {minutes} minutes, so it may be stuck"}
        return None


# ── managing watches ──────────────────────────────────────────────────────
def spawn(db, watch_id: str) -> int:
    """Start the watcher as its own process, so it outlives whoever asked for it."""
    import cortex
    logdir = os.path.expanduser("~/.cortex/logs")
    os.makedirs(logdir, exist_ok=True)
    kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": 0x00000208}
    with open(os.path.join(logdir, f"watch-{watch_id}.log"), "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "cortex.core.watchers", watch_id], cwd=cortex.ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
    db.watch_update(watch_id, pid=p.pid)
    return p.pid


def add(db, session_id: str, label: str, spec: dict) -> dict:
    """-> the watch, or {"error": why}."""
    problem = Probe(spec).problem()
    if problem:
        return {"error": problem}
    watch_id = uuid.uuid4().hex[:8]
    db.watch_add(watch_id, session_id, label, spec)
    spawn(db, watch_id)
    return {"id": watch_id, "label": label, "spec": spec}


def active(db, session_id: str = None) -> list:
    rows = db.watches_where("status = 'active'")
    return [w for w in rows if session_id is None or w["session_id"] == session_id]


def stop(db, watch_id: str) -> bool:
    w = db.watch_get(watch_id)
    if not w or w["status"] != "active":
        return False
    db.watch_update(watch_id, status="stopped")
    if w.get("pid"):
        try:
            os.kill(int(w["pid"]), 15)
        except OSError:
            pass
    return True


def revive(db) -> int:
    """After a reboot the watcher processes are gone but the requests still stand."""
    count = 0
    for w in active(db):
        if not (w.get("pid") and _pid_alive(w["pid"])):
            try:
                spawn(db, w["id"])
                count += 1
            except OSError as e:
                logger.warning(f"couldn't restart watch {w['id']}: {e}")
    return count


def plain_message(w: dict, event: dict) -> str:
    """What the user gets when the model can't be used."""
    text = f"{w['label']} {event['detail']}."
    if event.get("tail"):
        text += f"\n\nLast output:\n```\n{event['tail'][-1200:]}\n```"
    return text


def _prompt(w: dict, event: dict) -> str:
    from cortex.core.cortex_core import untrusted
    tail = untrusted("watched_process", event["tail"][-3000:]) if event.get("tail") else "(no output captured)"
    why = w["spec"].get("note") or "keep an eye on it"
    return (f"[Update from a watcher you started. Nobody typed this.] You were looking after: {w['label']}. "
            f"What the user asked: {why}.\nWhat just happened: it {event['detail']}.\nLatest output:\n{tail}\n\n"
            "Tell them now, in a couple of plain sentences: what happened and, if it went wrong, the likely "
            "cause from the output and what to do next. Only say it worked if the above shows that; if you "
            "can't tell, say so. Check something with a read-only tool only if you really need to, and "
            "do it before you reply: this message is all they'll get, so don't promise to look into it later.")


def deliver(db, w: dict, event: dict) -> None:
    """Come back to the chat with what happened."""
    import cortex.util.osutil as osutil
    reply = None
    try:
        import cortex.util.power as power
        if power.should_run("heavy")[0]:
            os.environ["CORTEX_BACKGROUND"] = "1"
            from cortex.core.cortex_core import CortexCore
            reply = CortexCore().report(w["session_id"], _prompt(w, event))
    except Exception as e:
        logger.warning(f"couldn't reach a model for the report: {e}")
    text = reply or plain_message(w, event)
    db.save_conversation(w["session_id"], "assistant", text, parts=[{"t": "watch", "label": w["label"]}])
    failed = event.get("code") not in (None, 0) or event["what"] == "stalled"
    db.add_nudge(w["label"][:80], text[:600], kind="warn" if failed else "info", source=f"chat:{w['session_id']}")
    osutil.notify(f"Cortex · {w['label'][:40]}", text[:200])


def run(watch_id: str) -> int:
    """The watcher process itself."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    from cortex.util.cortex_bootstrap import load_config, connect_db
    db = connect_db(load_config())
    w = db.watch_get(watch_id) if db else None
    if not w:
        return 1
    db.watch_update(watch_id, pid=os.getpid())
    spec = w["spec"]
    probe = Probe(spec)
    interval = spec.get("interval") or INTERVAL.get(spec["kind"], 10)
    expires = time.time() + spec.get("max_hours", MAX_HOURS) * 3600
    last_beat = time.time()
    while True:
        try:
            event = probe.step()
        except Exception as e:      # a flaky read shouldn't end the watch
            logger.warning(f"watch {watch_id}: {e}")
            event = None
        if event:
            current = db.watch_get(watch_id)
            if not current or current["status"] != "active":
                return 0
            deliver(db, w, event)
            if event.get("final"):
                db.watch_update(watch_id, status="done")
                return 0
        if time.time() > expires:
            deliver(db, w, {"what": "expired", "final": True, "tail": "",
                            "detail": f"has been running for over {spec.get('max_hours', MAX_HOURS)} hours, so I've stopped watching it"})
            db.watch_update(watch_id, status="expired")
            return 0
        if time.time() - last_beat > 30:
            last_beat = time.time()
            current = db.watch_get(watch_id)
            if not current or current["status"] != "active":
                return 0
            db.watch_update(watch_id)
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
