#!/usr/bin/env python3
"""
run_once.py — scheduled (wake-run-exit) entry point for Cortex.

Replaces the always-on KeepAlive Flask daemon for all background work. The tick
(cortex/core/tick.py) invokes one subcommand when it's due, which boots only what
it needs, does one pass, and exits. Recurring tasks have their own OS job.

  python3 run_once.py autonomy     # one multi-step autonomous session (heavy)
  python3 run_once.py reminders    # fire any due reminders (cheap)
  python3 run_once.py automation   # evaluate automation rules once (cheap)
  python3 run_once.py report NAME  # generate + deliver a scheduled report (heavy)

Flags:
  --force   ignore battery/thermal gating (run regardless)
  --dry-run build everything and report what WOULD run, but take no action
"""
from __future__ import annotations

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("cortex.run_once")

# Run from the project directory regardless of how launchd invokes us.
os.chdir(__import__("cortex").ROOT)

import cortex.util.power as power
from cortex.util.cortex_bootstrap import (load_config, connect_db, build_llm, build_for_autonomy,
                              ensure_state_dir, load_secrets)

# Load secrets (Keychain first, then the 0600 env file) so launchd can call this
# script via the venv python directly — no secret in the plist, no shell wrapper
# (which macOS TCC blocks inside ~/Documents). Only needed for Postgres sync;
# the default SQLite backend needs no secret at all.
load_secrets()

_CYCLE_COUNTER = os.path.expanduser("~/.cortex/state/autonomy_cycles.json")
PRIORITY_FORMATION_EVERY_N = 8


def _bump_cycle_counter() -> int:
    ensure_state_dir()
    n = 0
    try:
        with open(_CYCLE_COUNTER, encoding="utf-8") as f:
            n = int(json.load(f).get("count", 0))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        n = 0
    n += 1
    try:
        with open(_CYCLE_COUNTER, "w", encoding="utf-8") as f:
            json.dump({"count": n}, f)
    except Exception:
        pass
    return n


# ── subcommands ──────────────────────────────────────────────────────────
def cmd_autonomy(args) -> int:
    run, reason = (True, "forced") if args.force else power.should_run("heavy")
    logger.info(f"autonomy gate: run={run} ({reason})")
    if not run:
        return 0
    config = load_config()
    if args.dry_run:
        logger.info("dry-run: would build deps and run one autonomous session")
        return 0
    deps = build_for_autonomy(config)
    if deps is None:
        logger.warning("autonomy skipped: dependencies unavailable (DB down?)")
        return 0
    # Keep the local store bounded — the server's context_store had ballooned to
    # ~149k rows because purge never ran while the daemon was disabled.
    try:
        deps["db"].purge_expired_context()
    except Exception as e:
        logger.debug(f"context purge skipped: {e}")
    from cortex.core.autonomy_session import AutonomySession
    session = AutonomySession(deps["db"], deps["llm"], deps["executor"],
                              deps["context_reader"], deps["memory_mgr"],
                              config.get("autonomy", {}))
    result = session.run()
    logger.info(f"autonomy result: {result.get('status')}")
    if _bump_cycle_counter() % PRIORITY_FORMATION_EVERY_N == 0:
        logger.info("priority-formation cadence hit — reviewing recurring goals")
        try:
            session.maybe_form_priority()
        except Exception as e:
            logger.warning(f"priority formation failed: {e}")
    return 0


def cmd_reminders(args) -> int:
    run, reason = (True, "forced") if args.force else power.should_run("cheap")
    if not run:
        logger.info(f"reminders skipped ({reason})")
        return 0
    config = load_config()
    if args.dry_run:
        logger.info("dry-run: would fire due reminders")
        return 0
    db = connect_db(config)
    if db is None:
        return 0
    from cortex.core.reminders import ReminderManager
    ReminderManager(db, config.get("reminders", {})).run_once()
    return 0


def cmd_automation(args) -> int:
    run, reason = (True, "forced") if args.force else power.should_run("cheap")
    if not run:
        logger.info(f"automation skipped ({reason})")
        return 0
    config = load_config()
    if args.dry_run:
        logger.info("dry-run: would evaluate automation rules once")
        return 0
    db = connect_db(config)
    if db is None:
        return 0
    from cortex.core.automation import AutomationEngine
    fired = AutomationEngine(config.get("automation_rules", []), db=db).run_once()
    logger.info(f"automation: {fired} rule(s) fired")
    return 0


def cmd_report(args) -> int:
    run, reason = (True, "forced") if args.force else power.should_run("heavy", min_battery=20)
    if not run:
        logger.info(f"report skipped ({reason})")
        return 0
    config = load_config()
    if args.dry_run:
        logger.info(f"dry-run: would generate+deliver report '{args.name}'")
        return 0
    db = connect_db(config)
    if db is None:
        return 0
    from cortex.core.context_reader import ContextReader
    from cortex.core.memory import MemoryManager
    llm = build_llm(config)
    ctx = ContextReader(db, config.get("context", {}))
    mem = MemoryManager(db, llm, config.get("memory", {}))
    from cortex.core.scheduled_reports import ReportScheduler
    text = ReportScheduler(config).run_scheduled(args.name, llm, ctx, mem)
    logger.info(f"report delivered: {text[:120]}")
    return 0


def _notify(title: str, msg: str):
    """Fire a desktop notification (best-effort, cross-platform)."""
    import cortex.util.osutil as osutil
    osutil.notify(title, msg)


TASK_RULES = (
    "\n\n[Scheduled background run — nobody is reading along.] If there's nothing the user needs "
    "to hear right now, reply with exactly NOTHING_TO_REPORT. If this job has an end and it has now "
    "been reached (the thing they were waiting for happened), start your reply with TASK_COMPLETE "
    "and then say what happened. Otherwise reply with what they need to know."
)
TASK_MAX_FAILURES = 3


def _deliver_task(task: dict, text: str, db=None):
    """Put a task's result where the user will see it."""
    if db is not None:      # the bell, so it's still there after the toast is gone
        try:
            db.add_nudge(task["desc"][:80], text[:600], kind="info", source="task")
        except Exception as e:
            logger.debug(f"task nudge write skipped: {e}")
    deliver = task.get("deliver", "notify")
    if deliver in ("notify", "both"):
        _notify(f"Cortex · {task['desc'][:40]}", text)
    if deliver in ("email", "both"):
        try:
            import cortex.integrations.mail as mail, cortex.integrations.mail_store as mail_store, cortex.storage.settings_store as settings_store
            accts = settings_store.load().get("emails", [])
            if accts:
                frm = accts[0]["email"]
                pw = mail_store.password(frm)
                if pw:
                    ok, err = mail.send(frm, pw, frm, f"Cortex: {task['desc'][:60]}", text)
                    if not ok:
                        logger.warning(f"task email failed: {err}")
                else:
                    logger.warning(f"task email skipped: no app password for {frm}")
            else:
                logger.warning("task email delivery skipped: no account configured")
        except Exception as e:
            logger.warning(f"task email delivery failed: {e}")


def cmd_task(args) -> int:
    """Run one user-scheduled recurring task through Cortex and deliver what it found.
    A run with nothing to say stays silent, a finished job removes itself, and a task
    that can't reach a model pauses itself after a few tries instead of nagging forever."""
    import cortex.core.tasks as tasks_mod
    task = tasks_mod.get_task(args.name)
    if not task:
        logger.warning(f"task {args.name} not found — removing stale job")
        tasks_mod.remove_task(args.name)
        return 0
    run, reason = (True, "forced") if args.force else power.should_run("heavy")
    if not run:
        logger.info(f"task {task['id']} skipped ({reason})")
        return 0
    if args.dry_run:
        logger.info(f"dry-run: would run task '{task['desc']}'")
        return 0
    import cortex.core.cortex_core as cortex_core
    result = cortex_core.ask_full(task["prompt"] + TASK_RULES, session_id=f"task:{task['id']}", source="task")
    answer = (result.get("speak") or "").strip()
    logger.info(f"task {task['id']} -> {answer[:120]}")
    db = cortex_core._CORE.db if cortex_core._CORE else None

    if result.get("error"):
        fails = tasks_mod.record_failure(task["id"])
        logger.warning(f"task {task['id']} couldn't reach a model ({fails}/{TASK_MAX_FAILURES})")
        if fails >= TASK_MAX_FAILURES:
            tasks_mod.set_task_enabled(task["id"], False)
            _deliver_task({**task, "deliver": "notify"},
                          f"I've paused this because I couldn't reach a model {fails} times in a row. "
                          "Turn it back on in Settings → Background.", db)
        return 0
    tasks_mod.clear_failures(task["id"])

    if "TASK_COMPLETE" in answer:
        text = answer.replace("TASK_COMPLETE", "").strip(" :-—\n") or "All done."
        tasks_mod.remove_task(task["id"])
        _deliver_task(task, text, db)
        logger.info(f"task {task['id']} finished and removed itself")
        return 0
    if not answer or "NOTHING_TO_REPORT" in answer:
        return 0
    _deliver_task(task, answer, db)
    return 0


def cmd_sync(args) -> int:
    # Best-effort push of local SQLite -> remote Supabase. Self-skips when the
    # remote is unreachable (Tailscale off), so it's cheap to schedule.
    run, reason = (True, "forced") if args.force else power.should_run("cheap")
    if not run:
        logger.info(f"sync skipped ({reason})")
        return 0
    if args.dry_run:
        logger.info("dry-run: would push local changes to the remote if reachable")
        return 0
    import cortex.storage.sync as sync
    return sync.main()


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Cortex scheduled job runner")
    p.add_argument("--force", action="store_true", help="ignore battery/thermal gating")
    p.add_argument("--dry-run", action="store_true", help="build but take no action")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("autonomy")
    sub.add_parser("reminders")
    sub.add_parser("automation")
    sub.add_parser("sync")
    rp = sub.add_parser("report")
    rp.add_argument("name", nargs="?", default="Status")
    tp = sub.add_parser("task")
    tp.add_argument("name")
    args = p.parse_args()

    ensure_state_dir()
    # Scheduled runs are background work: tell the LLM layer to free the model's memory
    # almost immediately afterwards instead of leaving it resident for minutes.
    if args.cmd in ("autonomy", "report", "task"):
        os.environ["CORTEX_BACKGROUND"] = "1"
    if not args.force:
        import cortex.core.workers as workers
        allowed, why = workers.is_allowed(args.cmd, getattr(args, "name", None) if args.cmd in ("report", "task") else None)
        if not allowed:
            logger.info(f"{args.cmd} skipped ({why})")
            return 0
    return {
        "autonomy": cmd_autonomy,
        "reminders": cmd_reminders,
        "automation": cmd_automation,
        "sync": cmd_sync,
        "report": cmd_report,
        "task": cmd_task,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
