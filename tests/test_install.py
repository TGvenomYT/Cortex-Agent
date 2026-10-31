"""What a colleague's fresh machine depends on: scheduling, config, consent, portability."""
import json
import os
import plistlib
import subprocess
import sys

import pytest
import yaml

from cortex.core import scheduler, workers
from cortex.util import cortex_bootstrap as boot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVERY_HOUR = {"type": "interval", "seconds": 3600}


# ── scheduling on each OS ─────────────────────────────────────────────────
def test_launchd_job_is_valid_and_keeps_logs_out_of_the_project(monkeypatch):
    path = scheduler.write_plist("collector.test", ["/py", "/app/x.py"], EVERY_HOUR)
    job = plistlib.load(open(path, "rb"))
    assert job["Label"] == "com.cortex.collector.test" and job["StartInterval"] == 3600
    assert job["ProgramArguments"] == ["/py", "/app/x.py"] and "/.cortex/logs/" in job["StandardOutPath"]
    assert "EnvironmentVariables" in job


def test_launchd_daily_and_weekly():
    job = plistlib.load(open(scheduler.write_plist("a", ["x"], {"type": "weekly", "weekday": 5, "hour": 9, "minute": 30}), "rb"))
    assert job["StartCalendarInterval"] == {"Hour": 9, "Minute": 30, "Weekday": 5}


@pytest.mark.parametrize("schedule,want", [
    ({"type": "interval", "seconds": 300}, "*/5 * * * *"), ({"type": "interval", "seconds": 7200}, "0 */2 * * *"),
    ({"type": "interval", "seconds": 172800}, "0 0 */2 * *"), ({"type": "daily", "hour": 8, "minute": 15}, "15 8 * * *"),
    ({"type": "weekly", "weekday": 1, "hour": 9, "minute": 0}, "0 9 * * 1"),
])
def test_cron_times(schedule, want):
    assert scheduler.cron_when(schedule) == want


@pytest.fixture
def fake_crontab(monkeypatch):
    table = []
    monkeypatch.setattr(scheduler, "_cron_lines", lambda: list(table))

    def write(lines):
        table[:] = lines
    monkeypatch.setattr(scheduler, "_cron_write", write)
    monkeypatch.setattr(scheduler.osutil, "IS_MAC", False)
    monkeypatch.setattr(scheduler.osutil, "IS_LINUX", True)
    monkeypatch.setattr(scheduler.shutil, "which", lambda n: "/usr/bin/crontab")
    return table


def test_cron_job_adds_replaces_and_removes_only_its_own_line(fake_crontab):
    fake_crontab[:] = ["0 1 * * * backup.sh"]
    scheduler.install("reminders", ["/py", "/a b/run_once.py", "reminders"], {"type": "interval", "seconds": 60})
    scheduler.install("reminders", ["/py", "/a b/run_once.py", "reminders"], {"type": "interval", "seconds": 120})
    assert scheduler.installed("reminders")
    assert len([l for l in fake_crontab if "run_once" in l]) == 1 and "'/a b/run_once.py'" in fake_crontab[-1]
    scheduler.remove("reminders")
    assert fake_crontab == ["0 1 * * * backup.sh"] and not scheduler.installed("reminders")


@pytest.mark.parametrize("schedule,want", [
    ({"type": "interval", "seconds": 600}, ["/sc", "MINUTE", "/mo", "10"]),
    ({"type": "daily", "hour": 9, "minute": 0}, ["/sc", "DAILY", "/st", "09:00"]),
    ({"type": "weekly", "weekday": 5, "hour": 18, "minute": 30}, ["/sc", "WEEKLY", "/d", "FRI", "/st", "18:30"]),
])
def test_windows_task_arguments(schedule, want):
    args = scheduler.schtasks_args("collector.system", [r"C:\py.exe", r"C:\a b\x.py"], schedule)
    assert args[args.index("/tn") + 1] == r"\Cortex\collector-system" and args[-len(want):] == want
    assert '"C:\\a b\\x.py"' in args[args.index("/tr") + 1]


def test_workers_exist_and_switch_on_linux(fake_crontab, monkeypatch):
    monkeypatch.setattr(workers.osutil, "IS_MAC", False)
    rows = {r["key"]: r for r in workers.list_workers()}
    assert rows["reminders"]["available"] and not rows["reminders"]["on"]
    r = workers.set_enabled("reminders", True)
    assert r["on"] and workers.is_on("reminders"), r
    assert [l for l in fake_crontab if l.endswith("-tick")] and any("run_once.py" in l and "tick" in l for l in fake_crontab)
    workers.set_enabled("reminders", False)
    assert not workers.is_on("reminders") and fake_crontab == []


def test_every_worker_has_a_cadence_and_a_real_program():
    for w in workers.BUILTIN:
        assert "daily" in w or (w["every"] and w["ac"] and "battery" in w), w["key"]
        assert os.path.exists(os.path.join(ROOT, w["run"][0])), w["run"][0]


# ── consent ───────────────────────────────────────────────────────────────
def test_what_people_type_is_opt_in():
    defaults = {g["id"]: g["on"] for g in workers.TRACKING}
    assert defaults["shell"] is False and defaults["machine"] is False
    assert "collector.shell" not in workers.PROFILES["balanced"] and "collector.owner" not in workers.PROFILES["balanced"]


def test_choosing_turns_on_exactly_what_was_chosen():
    assert workers.choose_tracking(["essentials", "projects"])["ok"]
    assert workers.enabled_keys() == {"reminders", "automation", "collector.git", "collector.project"}


def test_nothing_ai_runs_without_being_asked():
    tracked = {k for g in workers.TRACKING for k in g["keys"]}
    assert not tracked & {"autonomy", "report-morning", "report-evening", "sync"}


# ── config ────────────────────────────────────────────────────────────────
def test_personal_config_sits_on_top_of_the_defaults(tmp_path, monkeypatch):
    shared, mine = tmp_path / "shared.yaml", tmp_path / "mine.yaml"
    shared.write_text(yaml.safe_dump({"llm": {"ollama": {"model": "a", "timeout": 90}}, "projects": []}))
    mine.write_text(yaml.safe_dump({"llm": {"ollama": {"model": "b"}}, "projects": ["x"]}))
    monkeypatch.setattr(boot, "CONFIG_PATHS", [str(shared), str(mine)])
    c = boot.load_config()
    assert c["llm"]["ollama"] == {"model": "b", "timeout": 90} and c["projects"] == ["x"]


def test_shared_config_has_nothing_personal():
    text = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    for word in ("100.100.", "rathinaswamy", "hotel_mapping", "/Users/", "Niranjan", "id_ed25519_myserver"):
        assert word not in text, word


@pytest.mark.parametrize("path", ["cortex", "collectors", "cx", "run.py", "uninstall.py", "smart_terminal_hook.zsh"])
def test_no_personal_details_in_the_shipped_code(path):
    full = os.path.join(ROOT, path)
    files = [full] if os.path.isfile(full) else [os.path.join(d, f) for d, _, fs in os.walk(full)
                                                 for f in fs if f.endswith(".py")]
    for f in files:
        text = open(f, encoding="utf-8").read()
        for word in ("/Users/niranjan", "rathinaswamy", "100.100.35.38", "id_ed25519_myserver", "hotel_mapping", "Niranjan"):
            assert word not in text, f"{word} in {f}"


# ── running from anywhere ─────────────────────────────────────────────────
def test_cx_finds_itself_through_a_symlink_from_any_folder(tmp_path):
    app = tmp_path / "some where" / "Cortex"
    (app / ".venv" / "bin").mkdir(parents=True)
    fake_python = app / ".venv" / "bin" / "python3"
    fake_python.write_text("#!/bin/sh\npwd\n")
    fake_python.chmod(0o755)
    launcher = app / "cx"
    launcher.write_text(open(os.path.join(ROOT, "cx")).read())
    launcher.chmod(0o755)
    link = tmp_path / "bin-cx"
    link.symlink_to(launcher)
    out = subprocess.run([str(link)], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0 and out.stdout.strip().endswith("some where/Cortex")


# ── other systems ─────────────────────────────────────────────────────────
def collector():
    sys.path.insert(0, os.path.join(ROOT, "collectors"))
    import system_collector
    return system_collector


def test_linux_memory_is_read_from_proc():
    total, free = collector().parse_meminfo("MemTotal: 16384000 kB\nMemFree: 1 kB\nMemAvailable: 8192000 kB\n")
    assert total == pytest.approx(16000) and free == pytest.approx(8000)


def test_the_collector_reports_no_public_ip_to_anyone():
    assert "ipify" not in open(os.path.join(ROOT, "collectors", "system_collector.py"), encoding="utf-8").read()


def test_liveness_never_uses_signal_zero_on_windows(monkeypatch):
    from cortex.core import runtime, watchers
    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(runtime.os, "kill", lambda *a: pytest.fail("os.kill on Windows sends Ctrl+C"))

    class Done:
        stdout = "python.exe  4242 Console"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    assert runtime._alive(4242) and watchers._pid_alive(4242) and not runtime._alive(99)
