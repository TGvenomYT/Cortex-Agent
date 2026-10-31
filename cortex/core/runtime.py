"""
runtime.py — keeps every running job accountable.

Each job has a saved record, so it can be stopped, watched, and picked up again after a
crash, instead of living only in a thread that dies with the process.

    Handle      one running job: its stop flag, what it's doing, when it last moved
    Runtime     starts and ends jobs, saves checkpoints, and runs the watchdog
    watchdog    a thread that notices a job that has gone quiet, and unblocks it

The agent loop can't stop itself while it's blocked on a slow model, so the watchdog does
it from outside by closing the connection, and the loop then decides what to do next.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid

logger = logging.getLogger("cortex.runtime")

ACTIVE = ("running",)


class JobStopped(Exception):
    """The job was stopped; the message says why."""


class ModelStalled(Exception):
    """The model has gone quiet for too long."""


class Handle:
    def __init__(self, job_id: str, session_id: str, source: str):
        self.id, self.session_id, self.source = job_id, session_id, source
        self.stop_flag = threading.Event()
        self.stall_flag = threading.Event()
        self.reason = ""
        self.phase = "starting"
        self.last_progress = time.monotonic()
        self.spent_usd, self.spent_tokens = 0.0, 0
        self._closers: list = []
        self._lock = threading.Lock()

    def touch(self, phase: str = None) -> None:
        self.last_progress = time.monotonic()
        if phase:
            self.phase = phase

    def idle(self) -> float:
        return time.monotonic() - self.last_progress

    def add_closer(self, close) -> None:
        """Register how to cut off whatever this job is blocked on (a model stream)."""
        with self._lock:
            self._closers.append(close)

    def release(self) -> None:
        with self._lock:
            closers, self._closers = self._closers, []
        for close in closers:
            try:
                close()
            except Exception as e:
                logger.debug(f"closing a stream: {e}")

    def stop(self, reason: str) -> None:
        self.reason = reason
        self.stop_flag.set()
        self.release()

    def check(self) -> None:
        """Called wherever the job can safely give way."""
        if self.stop_flag.is_set():
            raise JobStopped(self.reason or "stopped")
        if self.stall_flag.is_set():
            self.stall_flag.clear()
            raise ModelStalled()


class Runtime:
    def __init__(self, db, config: dict = None):
        cfg = config or {}
        self.db = db
        self.stall_seconds = cfg.get("stall_seconds", 150)
        self.beat_seconds = 10
        self._handles: dict[str, Handle] = {}     # session_id -> its running job
        self._lock = threading.Lock()
        self._watchdog = None

    # ── a job's life ──────────────────────────────────────────────────────
    def begin(self, job) -> Handle:
        h = Handle(job.id or uuid.uuid4().hex[:12], job.session_id, job.source)
        job.id = h.id
        with self._lock:
            self._handles[job.session_id] = h
        try:
            self.db.job_start(h.id, h.session_id, h.source, job.text, os.getpid())
        except Exception as e:
            logger.warning(f"job record not saved: {e}")
        self._ensure_watchdog()
        return h

    def save(self, h: Handle, messages: list, job) -> None:
        """Checkpoint a job after each step, so a crash loses at most the step in progress."""
        h.touch()
        try:
            self.db.job_save(h.id, job.step, {"messages": messages, "job": job.to_dict()})
        except Exception as e:
            logger.warning(f"job checkpoint not saved: {e}")

    def end(self, h: Handle, status: str, note: str = "") -> None:
        h.phase = "done"
        with self._lock:
            if self._handles.get(h.session_id) is h:
                del self._handles[h.session_id]
        try:
            self.db.job_end(h.id, status, note)
        except Exception as e:
            logger.warning(f"job end not saved: {e}")

    def stop(self, session_id: str, reason: str = "you stopped it") -> bool:
        with self._lock:
            h = self._handles.get(session_id)
        if h is None:
            return False
        h.stop(reason)
        return True

    def running(self, session_id: str) -> Handle | None:
        with self._lock:
            return self._handles.get(session_id)

    # ── after a crash ─────────────────────────────────────────────────────
    def recover(self) -> int:
        """Anything still marked running whose process is gone was cut off."""
        count = 0
        try:
            for j in self.db.jobs_with_status("running"):
                if j["pid"] != os.getpid() and _alive(j["pid"]):
                    continue
                if j["pid"] == os.getpid() and self.running(j["session_id"]):
                    continue
                self.db.job_end(j["id"], "interrupted", "Cortex stopped while this was running")
                count += 1
        except Exception as e:
            logger.warning(f"job recovery: {e}")
        return count

    # ── watchdog ──────────────────────────────────────────────────────────
    def _ensure_watchdog(self) -> None:
        if self._watchdog is None or not self._watchdog.is_alive():
            self._watchdog = threading.Thread(target=self._watch, name="cortex-watchdog", daemon=True)
            self._watchdog.start()

    def _watch(self) -> None:
        last_beat = 0.0
        while True:
            time.sleep(2)
            with self._lock:
                handles = list(self._handles.values())
            for h in handles:
                if h.phase == "model" and h.idle() > self.stall_seconds:
                    logger.warning(f"job {h.id}: model silent for {h.idle():.0f}s, cutting it off")
                    h.stall_flag.set()
                    h.release()
                    h.touch()
            if time.time() - last_beat >= self.beat_seconds:
                last_beat = time.time()
                for h in handles:
                    try:
                        self.db.job_beat(h.id)
                    except Exception as e:
                        logger.debug(f"job heartbeat: {e}")


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
