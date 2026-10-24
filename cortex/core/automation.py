"""
Automation Rules Engine — Condition → Action system.
Monitors system state and triggers actions automatically.

Examples:
- CPU > 80% for 5 min → notify
- Container crashes → restart it
- Disk > 90% → warn + suggest cleanup
- No git commit in 2 hours → nudge
"""

import os
import threading
import time
import json
import subprocess
import urllib.request
import logging
from datetime import datetime, timedelta

from cortex.util.notify import notify as _notify_os, say as _say_os

logger = logging.getLogger("cortex.automation")

BASE_URL = "http://localhost:7800"

# Cooldown state must survive across wake-run-exit invocations in the scheduled
# model, where the in-memory _fired_recently dict would reset every run and
# re-fire every rule each tick. Persist it to a small JSON file.
_COOLDOWN_STATE = os.path.expanduser("~/.cortex/state/automation_cooldowns.json")


class AutomationEngine:
    """Evaluates rules against current state and fires actions."""

    def __init__(self, rules: list, db=None):
        self.rules = rules
        self.db = db
        self._running = False
        self._thread = None
        self._fired_recently = {}  # rule_name → last_fired time (cooldown)

    def start(self):
        if not self.rules:
            logger.info("No automation rules configured")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="automation-engine")
        self._thread.start()
        logger.info(f"Automation engine started ({len(self.rules)} rules)")

    def stop(self):
        self._running = False

    # ── scheduled (wake-run-exit) entry point ────────────────────────────
    def run_once(self) -> int:
        """
        Evaluate every rule exactly once, with cooldown state loaded from and
        saved back to disk so cooldowns hold across separate launchd runs.
        Returns the number of rules that fired.
        """
        if not self.rules:
            return 0
        self._load_cooldowns()
        before = dict(self._fired_recently)
        for rule in self.rules:
            try:
                self._evaluate(rule)
            except Exception as e:
                logger.error(f"Rule evaluation failed ({rule.get('name', '?')}): {e}")
        fired = sum(1 for k, v in self._fired_recently.items() if before.get(k) != v)
        self._save_cooldowns()
        return fired

    def _load_cooldowns(self):
        try:
            with open(_COOLDOWN_STATE, encoding="utf-8") as f:
                raw = json.load(f)
            self._fired_recently = {k: datetime.fromisoformat(v) for k, v in raw.items()}
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            self._fired_recently = {}

    def _save_cooldowns(self):
        try:
            os.makedirs(os.path.dirname(_COOLDOWN_STATE), mode=0o700, exist_ok=True)
            with open(_COOLDOWN_STATE, "w", encoding="utf-8") as f:
                json.dump({k: v.isoformat() for k, v in self._fired_recently.items()}, f)
        except Exception as e:
            logger.warning(f"could not persist automation cooldowns: {e}")

    def _loop(self):
        while self._running:
            time.sleep(60)  # Evaluate every 60 seconds
            for rule in self.rules:
                try:
                    self._evaluate(rule)
                except Exception as e:
                    logger.error(f"Rule evaluation failed ({rule.get('name', '?')}): {e}")

    def _evaluate(self, rule: dict):
        """Evaluate a single rule."""
        name = rule.get("name", "unnamed")
        condition = rule.get("condition", {})
        action = rule.get("action", {})
        cooldown_min = rule.get("cooldown_minutes", 15)

        # Cooldown check
        last_fired = self._fired_recently.get(name)
        if last_fired and (datetime.now() - last_fired).total_seconds() < cooldown_min * 60:
            return

        # Evaluate condition
        if self._check_condition(condition):
            logger.info(f"Rule triggered: {name}")
            self._fire_action(name, action)
            self._fired_recently[name] = datetime.now()

    def _check_condition(self, condition: dict) -> bool:
        """Check if a condition is met."""
        ctype = condition.get("type", "")

        if ctype == "metric_threshold":
            return self._check_metric(condition)
        elif ctype == "container_status":
            return self._check_container(condition)
        elif ctype == "command":
            return self._check_command(condition)
        return False

    def _check_metric(self, condition: dict) -> bool:
        """Check metric from context_store."""
        if not self.db:
            return False
        metric = condition.get("metric", "")
        operator = condition.get("operator", ">")
        threshold = float(condition.get("threshold", 0))

        # Query from context_store
        rows = self.db.get_context(categories=["metrics", "system"], max_age_minutes=5)
        for row in rows:
            if row["key"] == metric:
                value = row["value"]
                if isinstance(value, dict):
                    actual = float(value.get("usage_pct", value.get("pct", 0)))
                else:
                    actual = float(value)

                if operator == ">" and actual > threshold:
                    return True
                elif operator == "<" and actual < threshold:
                    return True
                elif operator == ">=" and actual >= threshold:
                    return True
        return False

    def _check_container(self, condition: dict) -> bool:
        """Check if a container has a specific status."""
        container = condition.get("container", "")
        expected_status = condition.get("status", "exited")

        try:
            result = subprocess.run(
                f"docker inspect --format='{{{{.State.Status}}}}' {container}",
                shell=True, capture_output=True, text=True, timeout=5,
                env={"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
            )
            actual = result.stdout.strip()
            return actual == expected_status
        except Exception:
            return False

    def _check_command(self, condition: dict) -> bool:
        """Run a command and check exit code."""
        cmd = condition.get("command", "")
        expect_exit = condition.get("exit_code", 0)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=10,
                env={"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
            )
            return result.returncode == expect_exit
        except Exception:
            return False

    def _fire_action(self, rule_name: str, action: dict):
        """Execute the action for a triggered rule."""
        atype = action.get("type", "notify")

        if atype == "notify":
            message = action.get("message", f"Rule triggered: {rule_name}")
            self._notify(rule_name, message)

        elif atype == "say":
            message = action.get("message", f"Heads up Niranjan, {rule_name}")
            self._say(message)

        elif atype == "execute":
            cmd = action.get("command", "")
            if cmd:
                subprocess.run(
                    cmd, shell=True, capture_output=True, timeout=30,
                    env={"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
                )
                logger.info(f"Automation executed: {cmd}")

        elif atype == "restart_container":
            container = action.get("container", "")
            if container:
                subprocess.run(
                    f"docker restart {container}",
                    shell=True, capture_output=True, timeout=30,
                    env={"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
                )
                self._notify(rule_name, f"Auto-restarted container: {container}")
                logger.info(f"Auto-restarted: {container}")

    def _notify(self, title: str, message: str):
        _notify_os(title, message)

    def _say(self, text: str):
        _say_os(text)
