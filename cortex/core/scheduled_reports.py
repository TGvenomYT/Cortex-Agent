"""
Scheduled reports: launchd fires run_once at the report's time, and this builds the
briefing in-process and delivers it quietly.
"""

import json
import subprocess
import logging

logger = logging.getLogger("cortex.reports")

class ReportScheduler:
    """Generates and delivers one configured report per run."""

    def __init__(self, config: dict):
        self.reports = config.get("scheduled_reports", [])

    # ── scheduled (wake-run-exit) entry point ────────────────────────────
    def run_scheduled(self, name: str, llm, context_reader, memory_mgr=None) -> str:
        """
        Generate and deliver one named report IN-PROCESS (no HTTP server).
        launchd fires this at the report's clock time, so there is no polling
        loop and no "are you free?" retry dance — it just delivers, quietly.
        """
        report = next((r for r in self.reports
                       if r.get("name", "").lower() == (name or "").lower()), None)
        if report is None:
            report = {"name": name or "Status",
                      "prompt": "Give me a quick status update on everything.",
                      "delivery": "notify"}
        text = self._generate_inproc(report, llm, context_reader, memory_mgr)
        delivery = report.get("delivery", "notify")
        title = report.get("name", "Report")
        if delivery in ("notify", "both"):
            self._notify(title, text)
        if delivery in ("say", "both"):
            self._say(text)
        logger.info(f"delivered scheduled report '{title}' ({delivery})")
        return text

    def _generate_inproc(self, report: dict, llm, context_reader, memory_mgr=None) -> str:
        """Build the briefing locally from context + optional agentkit reads."""
        prompt = report.get("prompt", "Give me a status update.")
        context = ""
        try:
            context = context_reader.read_context()
        except Exception as e:
            logger.warning(f"context read failed: {e}")
        memory = ""
        if memory_mgr:
            try:
                memory = memory_mgr.format_for_prompt(limit=10)
            except Exception:
                pass
        from cortex.core.context_reader import ContextReader
        sys_prompt = (
            f"You are Cortex delivering a scheduled spoken briefing to {ContextReader.owner_name()}. "
            "Be concise and natural — a few sentences, no markdown. Use the live "
            "context below; call out anything that needs attention.\n\n"
            f"CONTEXT:\n{context}\n\nMEMORY:\n{memory}"
        )
        try:
            raw = llm.complete(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": prompt + " Respond as JSON: {\"speak\": \"...\"}"}],
                json_mode=True)
            return json.loads(raw).get("speak", "Here's your briefing.")
        except Exception as e:
            logger.error(f"in-process report generation failed: {e}")
            return "Couldn't generate the briefing right now."

    def _notify(self, title: str, text: str):
        safe_text = text[:200].replace('"', '\\"')
        script = f'display notification "{safe_text}" with title "Cortex: {title}" sound name "Ping"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)

    def _say(self, text: str):
        from cortex.llm.nlg import to_speech
        text = to_speech(text)
        subprocess.Popen(
            ["say", "-v", "Tara", "-r", "175", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
