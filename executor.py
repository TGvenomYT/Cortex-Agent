"""
Command Executor — Runs shell commands and logs results to DB.
This is the "hands" of Cortex.
"""

import os
import subprocess
import time
import logging

logger = logging.getLogger("cortex.executor")


class CommandExecutor:
    """Executes shell commands and logs everything."""

    def __init__(self, db):
        self.db = db

    def execute(self, command: str, cwd: str = None, timeout: int = 60, source: str = "api") -> dict:
        """
        Run a shell command.
        Returns: {exit_code, stdout, stderr, duration_ms}
        """
        logger.info(f"Executing [{source}]: {command}")
        start = time.time()

        # If cwd doesn't exist yet, run from home dir instead
        effective_cwd = cwd if cwd and os.path.isdir(cwd) else None

        try:
            # Ensure PATH includes common tool locations
            env = os.environ.copy()
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")

            result = subprocess.run(
                command,
                shell=True,
                cwd=effective_cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            duration_ms = int((time.time() - start) * 1000)

            output = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": duration_ms
            }

            # Log to DB
            self.db.log_command(
                command=command,
                exit_code=result.returncode,
                stdout=result.stdout[:5000],
                stderr=result.stderr[:2000],
                duration_ms=duration_ms,
                triggered_by=source,
                project_path=cwd
            )

            logger.info(f"Command finished: exit={result.returncode}, {duration_ms}ms")
            return output

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start) * 1000)
            output = {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "duration_ms": duration_ms
            }
            self.db.log_command(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                duration_ms=duration_ms,
                triggered_by=source,
                project_path=cwd
            )
            logger.warning(f"Command timed out: {command}")
            return output

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            output = {
                "exit_code": -2,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": duration_ms
            }
            self.db.log_command(
                command=command,
                exit_code=-2,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms,
                triggered_by=source,
                project_path=cwd
            )
            logger.error(f"Command failed with exception: {e}")
            return output
