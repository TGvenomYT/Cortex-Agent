"""
checkpoints.py — undo. Before Cortex changes files, it keeps a copy of what's there.

Works out which paths an action will touch, saves them under ~/.cortex/checkpoints, and
remembers paths that didn't exist yet, because undoing a new file means removing it. An
undo sets the current version aside first, so undoing is itself never destructive.

Be honest about the limit: a command whose effects can't be worked out from its text (a
script, `npm install`, anything on another machine) can't be undone, and the approval
prompt says so before you approve it.
"""
from __future__ import annotations

import logging
import os
import shutil
import time

logger = logging.getLogger("cortex.checkpoints")

ROOT = "~/.cortex/checkpoints"
MAX_BYTES = 200 * 1024 ** 2         # per checkpoint; a bigger target isn't copied, and that's reported
MAX_FILES = 5000
KEEP_DAYS = 14


def targets(action: dict, cwd: str = None) -> tuple[list, bool] | None:
    """-> (paths, covers_everything), or None if the action doesn't change files."""
    from cortex.core import safety
    kind = action.get("action")
    if kind == "edit":
        return [os.path.expanduser(action.get("edit_file") or "")], True
    if kind == "run_project":
        return [], False
    if kind in ("execute", "devops"):
        cmd = action.get("command") or ""
        if safety.classify_shell(cmd)[0] == "safe":
            return None
        return safety.write_targets(cmd, cwd)
    return None


def undo_note(action: dict) -> str:
    """One line for the approval prompt."""
    t = targets(action)
    if t is None:
        return ""
    paths, complete = t
    if complete and paths:
        return "\n\n*You can undo this afterwards.*"
    if paths:
        return "\n\n*I can undo part of this afterwards, not all of it.*"
    return "\n\n*I can't undo this one.*"


def _measure(path: str) -> tuple[int, int]:
    if os.path.islink(path) or os.path.isfile(path):
        return os.lstat(path).st_size, 1
    size = files = 0
    for folder, _, names in os.walk(path):
        for n in names:
            files += 1
            try:
                size += os.lstat(os.path.join(folder, n)).st_size
            except OSError:
                pass
        if files > MAX_FILES or size > MAX_BYTES:
            break
    return size, files


def _topmost_missing(path: str) -> str:
    """`mkdir -p a/b/c` creates a; undoing should remove a, not just c."""
    while True:
        parent = os.path.dirname(path)
        if not parent or parent == path or os.path.lexists(parent):
            return path
        path = parent


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src) and not os.path.islink(src):
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


class Checkpoints:
    def __init__(self, db, root: str = None):
        self.db = db
        self.root = os.path.expanduser(root or ROOT)

    def before(self, action: dict, job_id: str, session_id: str, label: str = "") -> dict | None:
        """Save what `action` is about to change. -> {id, complete} or None."""
        t = targets(action)
        if t is None:
            return None
        paths, complete = t
        items, used, seen = [], 0, set()
        cp_id = self.db.checkpoint_add(job_id, session_id, label or action.get("action", ""))
        folder = os.path.join(self.root, str(cp_id))
        for path in paths:
            path = os.path.abspath(path)
            if path in seen:
                continue
            seen.add(path)
            if not os.path.lexists(path):
                items.append({"path": _topmost_missing(path), "existed": False})
                continue
            size, files = _measure(path)
            if used + size > MAX_BYTES or files > MAX_FILES:
                complete = False
                logger.info(f"too big to keep a copy of: {path}")
                continue
            stored = os.path.join(folder, "files", str(len(items)))
            try:
                _copy(path, stored)
                used += size
                items.append({"path": path, "existed": True, "stored": stored})
            except OSError as e:
                complete = False
                logger.warning(f"couldn't copy {path}: {e}")
        self.db.checkpoint_set(cp_id, items, complete)
        return {"id": cp_id, "complete": complete, "count": len(items)}

    def summary(self, job_id: str) -> dict | None:
        rows = self.db.checkpoints_for_job(job_id)
        rows = [r for r in rows if r.get("items")]
        if not rows:
            return None
        return {"job": job_id, "count": sum(len(r["items"]) for r in rows),
                "complete": all(r.get("complete") for r in rows)}

    def undo_job(self, job_id: str) -> dict:
        """Put back everything a job changed, newest change first."""
        restored, removed, problems = [], [], []
        rows = self.db.checkpoints_for_job(job_id)
        if not rows:
            return {"ok": False, "message": "There's nothing to undo for that job."}
        complete = all(r.get("complete") for r in rows)
        for row in reversed(rows):
            aside = os.path.join(self.root, str(row["id"]), "replaced")
            for n, item in enumerate(reversed(row.get("items") or [])):
                path = item["path"]
                try:
                    if os.path.lexists(path):       # keep what's there, in case it's wanted
                        _copy(path, os.path.join(aside, str(n)))
                        _remove(path)
                    if item.get("existed"):
                        _copy(item["stored"], path)
                        restored.append(path)
                    else:
                        removed.append(path)
                except OSError as e:
                    problems.append(f"{path}: {e}")
            self.db.checkpoint_mark(row["id"], "undone")
        parts = []
        if restored:
            parts.append(f"put back {len(restored)} item{'s' if len(restored) != 1 else ''}")
        if removed:
            parts.append(f"removed {len(removed)} new item{'s' if len(removed) != 1 else ''}")
        message = ("Undone: " + " and ".join(parts) + ".") if parts else "Nothing needed changing."
        if not complete:
            message += " Some of what that job did couldn't be undone, so check the result."
        if problems:
            message += " These failed: " + "; ".join(problems[:3])
        return {"ok": not problems, "message": message, "restored": restored, "removed": removed}

    def undo_last(self, session_id: str) -> dict:
        job = self.db.last_undoable_job(session_id)
        if not job:
            return {"ok": False, "message": "There's nothing to undo in this chat."}
        return self.undo_job(job)

    def prune(self) -> int:
        """Drop copies older than KEEP_DAYS."""
        removed = 0
        cutoff = time.time() - KEEP_DAYS * 86400
        try:
            for name in os.listdir(self.root):
                path = os.path.join(self.root, name)
                if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
        except OSError:
            pass
        return removed
