"""
DevOps Agent — Multi-step task planning and execution with error recovery.
Plans tasks via LLM, executes step-by-step, feeds errors back to LLM for
recovery suggestions, supports rollback and live context gathering.
"""

import json
import uuid
import logging
import subprocess
from datetime import datetime

logger = logging.getLogger("cortex.devops")


class DevOpsAgent:
    """Plans and executes multi-step DevOps tasks with intelligence."""

    MAX_RETRIES = 2  # per step

    def __init__(self, db, llm, executor, context_reader=None):
        self.db = db
        self.llm = llm
        self.executor = executor
        self.context_reader = context_reader

    # ═══════════════════════════════════════════════════════════════
    # PLANNING
    # ═══════════════════════════════════════════════════════════════

    def create_plan(self, task: str, project_path: str = None) -> dict:
        """
        Generate an execution plan from a task description.
        Uses live context for the project if available.
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Gather live context
        context = self._gather_context(project_path)

        messages = [
            {"role": "system", "content": self._planning_prompt(context)},
            {"role": "user", "content": task}
        ]

        try:
            raw = self.llm.complete(messages, json_mode=True)
            data = json.loads(raw)
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return {
                "plan_id": plan_id,
                "speak": "I couldn't generate a plan for that task.",
                "plan": [],
                "error": str(e)
            }

        steps = data.get("plan", [])
        speak = data.get("speak", "Here's the plan.")
        rollback = data.get("rollback", [])

        # Store plan in DB
        self.db.save_plan(plan_id=plan_id, task=task, steps=steps, project_path=project_path)

        # Store rollback separately as metadata
        self._store_plan_metadata(plan_id, {
            "rollback": rollback,
            "task": task,
            "project_path": project_path,
            "created_at": datetime.now().isoformat(),
        })

        logger.info(f"Plan {plan_id}: {len(steps)} steps for '{task[:50]}'")
        return {
            "plan_id": plan_id,
            "speak": speak,
            "plan": steps,
            "rollback": rollback,
        }

    # ═══════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════

    def execute_plan(self, plan_id: str) -> dict:
        """
        Execute all steps sequentially.
        On failure: asks LLM for recovery (retry with fix, skip, or abort).
        """
        plan = self.db.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found", "status": "failed"}

        steps = plan["steps"] if isinstance(plan["steps"], list) else json.loads(plan["steps"])
        project_path = plan.get("project_path")

        self.db.update_plan_status(plan_id, "running")
        results = []

        for i, step in enumerate(steps):
            step_num = i + 1
            self.db.update_plan_status(plan_id, "running", current_step=step_num)
            logger.info(f"[{plan_id}] Step {step_num}/{len(steps)}: {step.get('desc', step.get('cmd', ''))}")

            # Execute the step
            result = self._execute_step(step, project_path, plan_id)

            if result["exit_code"] == 0:
                results.append(self._format_step_result(step, result, "success"))
                continue

            # ─── Step failed — attempt recovery ───
            logger.warning(f"[{plan_id}] Step {step_num} failed (exit {result['exit_code']})")

            recovery = self._attempt_recovery(
                plan_id=plan_id,
                step=step,
                step_num=step_num,
                error=result,
                project_path=project_path,
                remaining_steps=steps[i + 1:]
            )

            if recovery["action"] == "retry" and recovery.get("new_cmd"):
                # Try the recovery command
                retry_result = self.executor.execute(
                    command=recovery["new_cmd"],
                    cwd=project_path,
                    source="devops_agent"
                )
                if retry_result["exit_code"] == 0:
                    results.append(self._format_step_result(
                        step, retry_result, "recovered",
                        note=f"Original failed, recovered with: {recovery['new_cmd']}"
                    ))
                    continue
                else:
                    # Recovery also failed
                    results.append(self._format_step_result(step, result, "failed"))
                    self.db.update_plan_status(plan_id, "failed", current_step=step_num, results=results)
                    return {
                        "speak": f"Plan failed at step {step_num}: {step.get('desc', '')}. Recovery also failed.",
                        "results": results,
                        "status": "failed",
                        "failed_step": step_num,
                    }

            elif recovery["action"] == "skip":
                results.append(self._format_step_result(step, result, "skipped", note=recovery.get("reason", "")))
                logger.info(f"[{plan_id}] Skipping step {step_num} per LLM recommendation")
                continue

            else:  # abort
                results.append(self._format_step_result(step, result, "failed"))
                self.db.update_plan_status(plan_id, "failed", current_step=step_num, results=results)

                # Attempt rollback if available
                rollback_result = self._maybe_rollback(plan_id, project_path)

                return {
                    "speak": f"Plan aborted at step {step_num}: {step.get('desc', '')}. {recovery.get('reason', '')}",
                    "results": results,
                    "status": "failed",
                    "failed_step": step_num,
                    "rollback": rollback_result,
                }

        # All steps completed
        self.db.update_plan_status(plan_id, "completed", current_step=len(steps), results=results)
        logger.info(f"[{plan_id}] All {len(steps)} steps completed successfully")
        return {
            "speak": "All steps completed successfully.",
            "results": results,
            "status": "completed",
        }

    # ═══════════════════════════════════════════════════════════════
    # ERROR RECOVERY
    # ═══════════════════════════════════════════════════════════════

    def _attempt_recovery(self, plan_id: str, step: dict, step_num: int,
                          error: dict, project_path: str, remaining_steps: list) -> dict:
        """
        Ask LLM to analyze failure and suggest recovery.
        Returns: {action: retry|skip|abort, new_cmd: ..., reason: ...}
        """
        messages = [
            {"role": "system", "content": (
                "A DevOps step failed. Analyze the error and decide:\n"
                "Respond with JSON:\n"
                '{"action": "retry|skip|abort", "new_cmd": "fixed command or null", "reason": "brief explanation"}\n\n'
                "- retry: provide a fixed command that might work\n"
                "- skip: this step isn't critical, plan can continue\n"
                "- abort: this is a blocker, stop the plan\n\n"
                "Be practical. If a package isn't installed, install it. "
                "If a file doesn't exist, it might be in a different path. "
                "Don't retry the exact same command that failed."
            )},
            {"role": "user", "content": json.dumps({
                "step": step,
                "step_number": step_num,
                "exit_code": error["exit_code"],
                "stderr": error["stderr"][:500],
                "stdout": error["stdout"][:500],
                "remaining_steps": len(remaining_steps),
                "project_path": project_path,
            })}
        ]

        try:
            raw = self.llm.complete(messages, json_mode=True)
            recovery = json.loads(raw)
            logger.info(f"[{plan_id}] Recovery suggestion: {recovery.get('action')} — {recovery.get('reason', '')[:80]}")
            return recovery
        except Exception as e:
            logger.error(f"Recovery LLM call failed: {e}")
            return {"action": "abort", "reason": "Could not analyze error"}

    def _maybe_rollback(self, plan_id: str, project_path: str) -> dict:
        """Execute rollback steps if they were defined in the plan."""
        metadata = self._get_plan_metadata(plan_id)
        rollback_steps = metadata.get("rollback", []) if metadata else []

        if not rollback_steps:
            return {"attempted": False}

        logger.info(f"[{plan_id}] Executing {len(rollback_steps)} rollback steps")
        rollback_results = []
        for step in rollback_steps:
            cmd = step.get("cmd", step) if isinstance(step, dict) else step
            result = self.executor.execute(command=cmd, cwd=project_path, source="devops_rollback")
            rollback_results.append({
                "cmd": cmd,
                "exit_code": result["exit_code"],
                "stdout": result["stdout"][:500],
            })

        return {"attempted": True, "results": rollback_results}

    # ═══════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _execute_step(self, step: dict, project_path: str, plan_id: str) -> dict:
        """Execute a single step command."""
        cmd = step.get("cmd", "")
        timeout = step.get("timeout", 120)
        return self.executor.execute(
            command=cmd,
            cwd=project_path,
            timeout=timeout,
            source="devops_agent"
        )

    def _gather_context(self, project_path: str = None) -> str:
        """Gather live context for planning — richer than pre-computed."""
        parts = []

        # Pre-computed context from DB
        if self.context_reader:
            parts.append(self.context_reader.read_context(project_path=project_path))

        # Live additions for DevOps
        if project_path:
            # Current git state (live, not cached)
            git_status, _, _ = self._run_quick("git status --short", cwd=project_path)
            git_branch, _, _ = self._run_quick("git rev-parse --abbrev-ref HEAD", cwd=project_path)
            if git_status or git_branch:
                parts.append(f"[LIVE GIT]\n  branch: {git_branch}\n  status:\n{git_status}")

            # Docker state
            docker_ps, _, rc = self._run_quick("docker compose ps --format json 2>/dev/null", cwd=project_path)
            if rc == 0 and docker_ps:
                parts.append(f"[LIVE DOCKER]\n{docker_ps[:500]}")

        return "\n\n".join(parts) if parts else "(No context available)"

    def _run_quick(self, cmd: str, cwd: str = None) -> tuple:
        """Quick subprocess call for live context (not logged to DB)."""
        try:
            r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=5)
            return r.stdout.strip(), r.stderr.strip(), r.returncode
        except Exception:
            return "", "", -1

    def _planning_prompt(self, context: str) -> str:
        """Build the system prompt for plan generation."""
        return (
            "You are Cortex in DevOps planning mode. Create a precise execution plan.\n"
            "Respond with JSON:\n"
            "{\n"
            '  "speak": "1-2 sentence summary of the plan",\n'
            '  "plan": [\n'
            '    {"step": 1, "cmd": "exact shell command", "desc": "what this does", "critical": true/false}\n'
            "  ],\n"
            '  "rollback": ["cmd to undo if plan fails"]\n'
            "}\n\n"
            "Rules:\n"
            "- macOS with zsh. Use full paths where possible.\n"
            "- One atomic command per step (no && chains).\n"
            "- Mark steps as critical=true if failure should abort the plan.\n"
            "- Include verification steps (health checks, test runs) where appropriate.\n"
            "- Include rollback commands for destructive operations.\n"
            "- If deploying: pull → build → deploy → verify pattern.\n"
            "- Max 15 steps. If more needed, break into sub-plans.\n"
            f"\nCURRENT CONTEXT:\n{context}"
        )

    def _format_step_result(self, step: dict, result: dict, status: str, note: str = "") -> dict:
        """Format a step result for the response."""
        return {
            "step": step.get("step", 0),
            "cmd": step.get("cmd", ""),
            "desc": step.get("desc", ""),
            "status": status,
            "exit_code": result.get("exit_code", -1),
            "stdout": result.get("stdout", "")[:2000],
            "stderr": result.get("stderr", "")[:1000],
            "duration_ms": result.get("duration_ms", 0),
            "note": note,
        }

    def _store_plan_metadata(self, plan_id: str, metadata: dict):
        """Store extra plan metadata in context_store (piggyback on existing table)."""
        self.db.upsert_context(
            category="devops_meta",
            key=plan_id,
            value=metadata,
            expires_minutes=1440  # 24 hours
        )

    def _get_plan_metadata(self, plan_id: str) -> dict:
        """Retrieve plan metadata."""
        rows = self.db.get_context(categories=["devops_meta"], max_age_minutes=1440)
        for row in rows:
            if row["key"] == plan_id:
                return row["value"]
        return {}
