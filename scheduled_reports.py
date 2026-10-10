"""
Scheduled Reports — Conversational delivery.
Asks if you're free before delivering. Retries if no response.
Reschedules if you say "later".
"""

import threading
import time
import json
import subprocess
import urllib.request
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("cortex.reports")

BASE_URL = "http://localhost:7800"


class ReportScheduler:
    """Conversational report scheduler — asks before delivering."""

    def __init__(self, config: dict):
        self.reports = config.get("scheduled_reports", [])
        self._running = False
        self._thread = None
        self._last_run = {}
        self._pending = []  # Reports waiting for user response
        self._retry_count = {}

    def start(self):
        if not self.reports:
            logger.info("No scheduled reports configured")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="report-scheduler")
        self._thread.start()
        logger.info(f"Report scheduler started ({len(self.reports)} reports)")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            now = datetime.now()

            # Check scheduled reports
            for report in self.reports:
                report_id = report.get("name", "unnamed")
                schedule_hour = report.get("hour", 9)
                schedule_minute = report.get("minute", 0)

                if now.hour == schedule_hour and now.minute == schedule_minute:
                    last = self._last_run.get(report_id)
                    if not last or (now - last).total_seconds() > 3600:
                        self._last_run[report_id] = now
                        self._initiate_report(report)

            # Check pending retries
            self._check_pending()

            time.sleep(30)

    def _initiate_report(self, report: dict):
        """Start the conversational delivery — ask if user is free."""
        name = report.get("name", "Report")
        logger.info(f"Initiating report: {name}")

        # Ask if they're free
        self._ask_availability(report)

    def _ask_availability(self, report: dict):
        """Ask the user if they're available for the report."""
        name = report.get("name", "Report")

        # Notify + speak the ask
        ask_text = f"Hey Niranjan, I've got your {name.lower()} ready. You free to hear it?"

        self._notify(name, ask_text)
        self._say(ask_text)

        # Add to pending — will check for response or retry
        self._pending.append({
            "report": report,
            "asked_at": datetime.now(),
            "retries": 0,
        })

    def _check_pending(self):
        """Check pending reports — retry after 5 min if no response."""
        now = datetime.now()
        still_pending = []

        for item in self._pending:
            report = item["report"]
            name = report.get("name", "")
            asked_at = item["asked_at"]
            retries = item["retries"]
            elapsed = (now - asked_at).total_seconds()

            if retries >= 3:
                # Give up after 3 retries — just leave the notification
                logger.info(f"Report {name}: gave up after 3 retries")
                self._notify(name, f"Your {name.lower()} is ready whenever you want it. Just ask.")
                continue

            if elapsed >= 300:  # 5 minutes since last ask
                # Retry
                item["retries"] += 1
                item["asked_at"] = now
                ask_text = f"Niranjan, still got that {name.lower()} for you. Want to hear it now?"
                self._say(ask_text)
                self._notify(name, ask_text)
                still_pending.append(item)
                logger.info(f"Report {name}: retry {item['retries']}")
            else:
                still_pending.append(item)

        self._pending = still_pending

    def deliver_report(self, report_name: str = None):
        """
        Actually deliver the report (called when user says yes).
        If report_name is None, deliver the first pending one.
        """
        if not self._pending and not report_name:
            return "No pending reports right now."

        # Find the report to deliver
        report = None
        for i, item in enumerate(self._pending):
            if report_name is None or item["report"].get("name", "").lower() == report_name.lower():
                report = item["report"]
                self._pending.pop(i)
                break

        if not report:
            # Generate on demand
            report = {"name": report_name or "Status", "prompt": "Give me a quick status update on everything."}

        return self._generate_report(report)

    def reschedule(self, minutes: int = 30, report_name: str = None):
        """Reschedule a pending report for later."""
        for i, item in enumerate(self._pending):
            if report_name is None or item["report"].get("name", "").lower() == report_name.lower():
                # Remove from pending, re-add with future time
                report = item["report"]
                self._pending.pop(i)
                logger.info(f"Report {report.get('name')}: rescheduled for {minutes} min")

                # Schedule a one-time delivery
                def _delayed_deliver():
                    time.sleep(minutes * 60)
                    self._initiate_report(report)

                threading.Thread(target=_delayed_deliver, daemon=True).start()
                return f"Alright, I'll ask again in {minutes} minutes."

        return "No pending report to reschedule."

    def _generate_report(self, report: dict) -> str:
        """Generate the report content via Cortex."""
        prompt = report.get("prompt", "Give me a status update.")
        name = report.get("name", "Report")

        logger.info(f"Generating report: {name}")

        try:
            data = json.dumps({
                "text": prompt,
                "source": "siri",
                "session_id": "reports"
            }).encode()
            req = urllib.request.Request(
                f"{BASE_URL}/ask",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())

            return body.get("speak", "Couldn't generate the report.")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return "Something went wrong generating the report."

    def has_pending(self) -> bool:
        """Check if there are pending reports waiting."""
        return len(self._pending) > 0

    def get_pending_names(self) -> list:
        """Get names of pending reports."""
        return [item["report"].get("name", "Report") for item in self._pending]

    def _notify(self, title: str, text: str):
        safe_text = text[:200].replace('"', '\\"')
        script = f'display notification "{safe_text}" with title "Cortex: {title}" sound name "Ping"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)

    def _say(self, text: str):
        from nlg import to_speech
        text = to_speech(text)
        subprocess.Popen(
            ["say", "-v", "Tara", "-r", "175", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
