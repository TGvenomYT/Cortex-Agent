"""
Reminder System — Natural language reminders with scheduling.
Runs as a background thread inside the daemon.
Fires via macOS notifications (osascript) + TTS (say command).
"""

import time
import threading
import subprocess
import logging
from datetime import datetime

logger = logging.getLogger("cortex.reminders")


class ReminderManager:
    """Background reminder checker — fires due reminders via notification + voice."""

    def __init__(self, db, config: dict):
        self.db = db
        self.check_interval = config.get("check_interval_seconds", 30)
        self.notification_method = config.get("notification_method", "both")
        self._thread = None
        self._running = False

    def start(self):
        """Start the background reminder checker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True, name="reminder-checker")
        self._thread.start()
        logger.info(f"Reminder checker started (interval={self.check_interval}s)")

    def stop(self):
        """Stop the checker thread."""
        self._running = False
        logger.info("Reminder checker stopped")

    def _check_loop(self):
        """Background loop: check for due reminders every N seconds."""
        while self._running:
            try:
                self._fire_due_reminders()
            except Exception as e:
                logger.error(f"Reminder check error: {e}")
            time.sleep(self.check_interval)

    def _fire_due_reminders(self):
        """Check DB for due reminders and fire them."""
        try:
            pending = self.db.get_pending_reminders()
        except Exception as e:
            logger.error(f"Failed to fetch reminders: {e}")
            return

        for reminder in pending:
            try:
                self.fire_reminder(dict(reminder))
                self.db.mark_reminder_fired(reminder["id"])

                # Handle recurring reminders
                repeat = reminder.get("repeat_rule")
                if repeat:
                    self._reschedule(reminder)

            except Exception as e:
                logger.error(f"Failed to fire reminder {reminder['id']}: {e}")

    def fire_reminder(self, reminder: dict):
        """Fire a reminder: macOS notification + natural TTS."""
        text = reminder.get("text", "Reminder")
        logger.info(f"Firing reminder: {text}")

        # Generate a natural spoken version
        natural_text = self._make_natural(text)

        method = self.notification_method

        if method in ("macos_notification", "both"):
            self._notify(text, natural_text)

        if method in ("say", "both"):
            self._say(natural_text)

    def _make_natural(self, text: str) -> str:
        """Convert a raw reminder into natural conversational speech."""
        # Simple natural phrases — no LLM call needed for speed
        import random
        intros = [
            f"Hey, quick reminder — {text}.",
            f"Heads up Niranjan, you wanted to {text}.",
            f"Just nudging you — {text}.",
            f"Hey, it's time to {text}.",
            f"Niranjan, reminder: {text}.",
        ]
        return random.choice(intros)

    def _notify(self, text: str, natural_text: str):
        """Send a macOS notification via osascript."""
        safe_text = natural_text.replace('"', '\\"').replace("'", "\\'")
        script = f'display notification "{safe_text}" with title "Cortex" subtitle "Reminder" sound name "Ping"'
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, timeout=5
            )
        except Exception as e:
            logger.error(f"Notification failed: {e}")

    def _say(self, text: str):
        """Speak naturally using macOS TTS."""
        try:
            subprocess.Popen(
                ["say", "-v", "Samantha", "-r", "180", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    def _reschedule(self, reminder: dict):
        """Reschedule a recurring reminder after it fires."""
        from datetime import timedelta
        rule = reminder.get("repeat_rule", "")
        due = reminder.get("due_at")
        if not due:
            return

        if isinstance(due, str):
            due = datetime.fromisoformat(due)

        next_due = None
        if rule == "daily":
            next_due = due + timedelta(days=1)
        elif rule == "weekly":
            next_due = due + timedelta(weeks=1)
        elif rule == "weekdays":
            next_due = due + timedelta(days=1)
            # Skip to Monday if landing on weekend
            while next_due.weekday() >= 5:
                next_due += timedelta(days=1)

        if next_due:
            try:
                self.db.create_reminder(
                    text=reminder["text"],
                    due_at=next_due,
                    repeat_rule=rule
                )
                logger.info(f"Rescheduled '{reminder['text']}' → {next_due.isoformat()}")
            except Exception as e:
                logger.error(f"Reschedule failed: {e}")

    def get_active(self) -> list:
        """Get all unfired reminders."""
        with self.db.get_conn() as conn:
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, text, due_at, repeat_rule, created_at
                    FROM reminders
                    WHERE fired = FALSE
                    ORDER BY due_at ASC
                """)
                rows = cur.fetchall()
        return [dict(r) for r in rows]
