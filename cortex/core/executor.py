"""
Command Executor — Runs shell commands and logs results to DB.
This is the "hands" of Cortex.

Each command gets its own process group, so stopping it stops everything it started, and
its output goes to files, so a long command can be handed to a watcher instead of killed.
"""

import os
import re
import signal
import subprocess
import tempfile
import time
import logging
import uuid

from cortex.core.safety import redact

logger = logging.getLogger("cortex.executor")

RUNS_DIR = "~/.cortex/runs"
_QUOTED_HOME = re.compile(r"""(?<![\w/\\])(["'])~/""")


def _home_in_quotes(command: str) -> str:
    """The model rightly quotes a path with a space in it, as "~/My Folder", but a shell only
    expands ~ outside quotes, so the folder looks like it isn't there."""
    return _QUOTED_HOME.sub(lambda m: '"$HOME/' if m[1] == '"' else "\"$HOME\"'/", command)


def _kill_group(proc) -> None:
    """Stop a command and all its children, politely first."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            return
        os.killpg(proc.pid, signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def prune_runs(max_age_days: int = 3) -> None:
    """Output files of finished background commands aren't needed for long."""
    folder = os.path.expanduser(RUNS_DIR)
    cutoff = time.time() - max_age_days * 86400
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
    except OSError:
        pass


class CommandExecutor:
    """Executes shell commands and logs everything."""

    def __init__(self, db):
        self.db = db

    def execute(self, command: str, cwd: str = None, timeout: int = 60, source: str = "api",
                should_stop=None, detach_after: float = None) -> dict:
        """
        Run a shell command.
        Returns: {exit_code, stdout, stderr, duration_ms}

        should_stop: called while it runs; returning True kills it ("stopped": True).
        detach_after: seconds before a command still running is left to carry on by itself.
          Returns straight away with "detached": {pid, log, exit_file}, for a watcher to follow.
        """
        logger.info(f"Executing [{source}]: {command}")
        start = time.time()

        # If cwd doesn't exist yet, run from home dir instead
        effective_cwd = cwd if cwd and os.path.isdir(cwd) else None

        from cortex.llm.viz import clean_env
        env = clean_env()   # commands never inherit Cortex's own passwords or API keys
        # Ensure PATH includes common tool locations
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")

        run_id = uuid.uuid4().hex[:10]
        folder = os.path.expanduser(RUNS_DIR)
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            folder = tempfile.gettempdir()
        out_path = os.path.join(folder, f"{run_id}.log")
        err_path = os.path.join(folder, f"{run_id}.err")
        code_path = os.path.join(folder, f"{run_id}.code")

        wrapped = command
        if os.name != "nt":   # the exit code has to outlive us, in case this is left running
            wrapped = f"(\n{_home_in_quotes(command)}\n)\necho $? > {code_path!r}"

        def finish(exit_code, stdout, stderr, **extra):
            duration_ms = int((time.time() - start) * 1000)
            self.db.log_command(command=redact(command), exit_code=exit_code, stdout=redact(stdout or "")[:5000],
                                stderr=redact(stderr or "")[:2000], duration_ms=duration_ms,
                                triggered_by=source, project_path=cwd)
            logger.info(f"Command finished: exit={exit_code}, {duration_ms}ms")
            return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
                    "duration_ms": duration_ms, **extra}

        def read(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                return ""

        def cleanup():
            for p in (out_path, err_path, code_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        group = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        try:
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                proc = subprocess.Popen(wrapped, shell=True, cwd=effective_cwd, stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, env=env, **group)
        except Exception as e:
            cleanup()
            logger.error(f"Command failed with exception: {e}")
            return finish(-2, "", str(e))

        pause = 0.01
        while proc.poll() is None:
            elapsed = time.time() - start
            if should_stop and should_stop():
                _kill_group(proc)
                stdout, stderr = read(out_path), read(err_path)
                cleanup()
                return finish(-4, stdout, stderr or "stopped", stopped=True)
            if detach_after is not None and elapsed >= detach_after:
                # Nothing else will collect it if this process lives on, and an uncollected
                # child looks alive forever to anyone watching.
                import threading
                threading.Thread(target=proc.wait, daemon=True).start()
                return finish(-3, read(out_path)[-2000:], "", detached={
                    "pid": proc.pid, "log": out_path, "err_log": err_path, "exit_file": code_path})
            if elapsed >= timeout:
                _kill_group(proc)
                stdout = read(out_path)
                cleanup()
                logger.warning(f"Command timed out: {command}")
                return finish(-1, stdout, f"Command timed out after {timeout}s")
            time.sleep(pause)
            pause = min(0.2, pause * 1.5)

        stdout, stderr = read(out_path), read(err_path)
        code = proc.returncode
        try:
            code = int(read(code_path).strip())
        except ValueError:
            pass
        cleanup()
        return finish(code, stdout, stderr)
