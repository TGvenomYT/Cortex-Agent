"""Background work: one job, what runs when, reminders on the second, and moving off the old jobs."""
import os
import plistlib
import subprocess
import time
from datetime import datetime, timedelta

import pytest

from cortex.core import alarm, automation, reminders, scheduler, tick, workers
import cortex.storage.settings_store as ss
from cortex.storage.db import Database
from cortex.util import power

BATTERY = {"on_ac": False, "percent": 70, "charging": False, "has_battery": True}
PLUGGED = {"on_ac": True, "percent": 70, "charging": True, "has_battery": True}
DESKTOP = {"on_ac": True, "percent": None, "charging": False, "has_battery": False}
ALL = {w["key"] for w in workers.BUILTIN}
NOON = datetime(2026, 11, 2, 12, 0).timestamp()


@pytest.fixture(autouse=True)
def cool(monkeypatch):
    monkeypatch.setattr(power, "thermal_pressure", lambda: "nominal")


@pytest.fixture
def due(monkeypatch):
    def run(batt, last=None, keys=ALL, now=NOON):
        monkeypatch.setattr(power, "battery_state", lambda: batt)
        state = {"last": {k: 0 for k in ALL} if last is None else last, "exit": {}}
        return [w["key"] for w in tick.due_now(keys, state, now=now, batt=batt)]
    return run


# ── what runs when ────────────────────────────────────────────────────────
def test_battery_runs_only_reminders_and_rules(due):
    assert due(BATTERY) == ["reminders", "automation"]


def test_plugged_in_laptop_runs_things_but_far_less_often(due):
    just_now = {k: NOON - 700 for k in ALL}
    assert due(PLUGGED, just_now) == ["automation"]
    half_hour = {k: NOON - 1800 for k in ALL}
    assert set(due(PLUGGED, half_hour)) >= {"collector.system", "collector.process", "collector.git"}
    assert "collector.owner" not in due(PLUGGED, half_hour) and "autonomy" not in due(PLUGGED, half_hour)


def test_a_server_keeps_the_original_cadences(due):
    three_minutes = {k: NOON - 180 for k in ALL}
    assert set(due(DESKTOP, three_minutes)) == {"reminders", "automation", "collector.process",
                                                "collector.metrics", "collector.shell"}


def test_a_battery_means_laptop_and_saying_otherwise_wins():
    assert workers.machine_kind(BATTERY) == "laptop" and workers.machine_kind(DESKTOP) == "server"
    ss.update_section("resources", {"machine": "server"})
    assert workers.power_mode(BATTERY) == "server"
    ss.update_section("resources", {"machine": "laptop"})
    assert workers.power_mode(DESKTOP) == "plugged"


def test_the_whole_point_wakes_an_hour():
    before = sum(3600 / w["every"] for w in workers.BUILTIN if "every" in w)
    assert before > 180 and 3600 / workers.TICK["laptop"] == 6


def test_one_model_load_a_tick(due):
    assert [k for k in due(PLUGGED) if k in ("autonomy", "report-morning", "report-evening")] == ["autonomy"]


def test_ai_on_battery_is_the_users_call(due):
    assert due(BATTERY, keys={"autonomy"}) == []
    ss.update_section("resources", {"ai_jobs_only_on_ac": False})
    assert due(BATTERY, keys={"autonomy"}) == ["autonomy"]


def test_a_hot_machine_runs_nothing(due, monkeypatch):
    monkeypatch.setattr(power, "thermal_pressure", lambda: "critical")
    assert due(PLUGGED) == []


def test_a_briefing_missed_on_battery_runs_once_plugged_in():
    w = next(x for x in workers.BUILTIN if x["key"] == "report-morning")
    nine = datetime(2026, 11, 2, 9, 0).timestamp()
    assert not tick.is_due(w, 0, nine - 60, "plugged")                  # not yet
    assert tick.is_due(w, nine - 86400, nine + 3600, "plugged")         # three hours late is fine
    assert not tick.is_due(w, nine + 10, nine + 3600, "plugged")        # once a day
    assert not tick.is_due(w, nine - 86400, nine + tick.GRACE + 60, "plugged")     # too late to matter


def test_work_skipped_on_battery_catches_up_when_plugged_in(due):
    last = {"collector.git": NOON - 7200}
    assert due(BATTERY, last, keys={"collector.git"}) == []
    assert due(PLUGGED, last, keys={"collector.git"}) == ["collector.git"]


def test_a_little_early_still_counts():
    w = next(x for x in workers.BUILTIN if x["key"] == "collector.git")
    assert tick.is_due(w, NOON - 1790, NOON, "plugged")
    git_server = NOON - 299
    assert tick.is_due(w, git_server, NOON, "server")


# ── the tick itself ───────────────────────────────────────────────────────
def test_a_run_records_what_happened(monkeypatch, tmp_path):
    w = {"key": "x", "ai": False}
    monkeypatch.setattr(workers, "command", lambda _: ["/bin/sh", "-c", "exit 3"])
    state = tick.load_state()
    tick.run(w, state)
    saved = tick.load_state()
    assert saved["exit"]["x"] == 3 and saved["last"]["x"] > 0 and "running" not in saved


def test_a_stuck_worker_gives_way(monkeypatch):
    monkeypatch.setattr(tick, "TIMEOUT", 0.2)
    monkeypatch.setattr(workers, "command", lambda _: ["/bin/sleep", "5"])
    tick.run({"key": "x", "ai": False}, tick.load_state())
    assert tick.load_state()["exit"]["x"] == 124


def test_nothing_on_does_nothing(monkeypatch):
    monkeypatch.setattr(tick, "due_now", lambda *a: pytest.fail("shouldn't look"))
    assert tick.main() == 0 and not os.path.exists(tick._path("tick.lock"))


def test_only_one_at_a_time_but_a_dead_lock_clears(monkeypatch):
    assert tick._lock() and not tick._lock()
    old = time.time() - tick.AI_TIMEOUT - 120
    os.utime(tick._path("tick.lock"), (old, old))
    assert tick._lock()


def test_it_stays_light():
    code = ("import sys; sys.path.insert(0, %r); import cortex.core.tick; "
            "print([m for m in ('yaml', 'psycopg2', 'httpx', 'numpy', 'flask') if m in sys.modules])"
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([os.sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stderr


# ── on demand ─────────────────────────────────────────────────────────────
def test_opening_cortex_refreshes_only_what_was_agreed_to():
    keys = {"collector.git", "collector.system", "collector.owner", "reminders"}
    assert {w["key"] for w in tick.stale(keys, {"last": {}})} == {"collector.git", "collector.system"}


def test_no_refresh_when_things_are_fresh(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "detach", lambda a, n: calls.append(a))
    ss.save({"workers_on": ["collector.git"]})
    assert tick.refresh_context() and "--context" in calls[0]
    tick.stamp(["collector.git"])
    assert not tick.refresh_context() and len(calls) == 1


# ── reminders ─────────────────────────────────────────────────────────────
@pytest.fixture
def db(tmp_path):
    d = Database(backend="sqlite", sqlite_path=str(tmp_path / "t.db"))
    assert d.connect()
    os.makedirs(os.path.expanduser("~/.cortex/state"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.cortex/logs"), exist_ok=True)
    return d


@pytest.fixture
def spawned(monkeypatch):
    calls = []
    monkeypatch.setattr(scheduler, "detach", lambda a, n: calls.append(n))
    ss.save({"workers_on": ["reminders"]})
    return calls


def in_(seconds):
    return datetime.now() + timedelta(seconds=seconds)


def test_every_change_moves_the_alarm(db, spawned):
    far = db.create_reminder("later", in_(7200))
    assert abs(alarm.next_due() - in_(7200).timestamp()) < 2 and not spawned
    near = db.create_reminder("soon", in_(120))
    assert abs(alarm.next_due() - in_(120).timestamp()) < 2 and spawned == ["alarm"]
    db.update_reminder(near, due_at=in_(3600))
    assert abs(alarm.next_due() - in_(3600).timestamp()) < 2
    db.delete_reminder(near)
    db.delete_reminder(far)
    assert alarm.next_due() is None


def test_nothing_sleeps_for_a_reminder_days_away(db, spawned):
    db.create_reminder("next week", in_(7 * 86400))
    alarm.check()
    assert not spawned


def test_the_tick_starts_the_sleeper_as_the_time_nears(db, spawned, monkeypatch):
    db.create_reminder("soon-ish", in_(alarm.HORIZON + 60))
    assert not spawned
    monkeypatch.setattr(alarm.time, "time", lambda real=time.time: real() + 120)
    alarm.check()
    assert spawned == ["alarm"]


def test_a_slow_tick_cannot_miss_the_window():
    assert alarm.HORIZON > max(workers.TICK.values())


def test_switched_off_means_no_alarm(db, spawned):
    ss.save({"workers_on": []})
    db.create_reminder("soon", in_(60))
    assert not spawned


def test_the_sleeper_rings_on_time_and_leaves(db, spawned, monkeypatch):
    db.create_reminder("soon", in_(1))
    rung = []

    def ring():
        rung.append(time.time())
        reminders.ReminderManager(db, {"notification_method": "none"}).run_once()
    monkeypatch.setattr(alarm, "_ring", ring)
    due = alarm.next_due()
    assert alarm.main() == 0
    assert len(rung) == 1 and abs(rung[0] - due) < 1 and not os.path.exists(alarm._path("alarm.beat"))


def test_the_sleeper_doesnt_spin_on_one_it_cant_clear(db, spawned, monkeypatch):
    db.create_reminder("stuck", in_(-5))
    rung = []
    monkeypatch.setattr(alarm, "_ring", lambda: rung.append(1))
    assert alarm.main() == 0 and len(rung) == 1


def test_the_sleeper_stops_if_reminders_get_switched_off(db, spawned, monkeypatch):
    db.create_reminder("soon", in_(600))
    naps = []

    def nap(_):
        naps.append(1)
        ss.save({"workers_on": []})
    monkeypatch.setattr(alarm.time, "sleep", nap)
    assert alarm.main() == 0 and naps


def test_a_second_sleeper_stands_down(db, spawned, monkeypatch):
    alarm._beat()
    monkeypatch.setattr(alarm, "next_due", lambda: pytest.fail("should have left"))
    assert alarm.main() == 0


def test_a_dead_sleeper_is_noticed(db, spawned):
    alarm._beat()
    old = time.time() - 300
    os.utime(alarm._path("alarm.beat"), (old, old))
    db.create_reminder("soon", in_(30))
    assert spawned == ["alarm"]


def test_two_ringers_cannot_ring_the_same_reminder(db, spawned):
    rid = db.create_reminder("once", in_(-1))
    assert db.mark_reminder_fired(rid) is True and db.mark_reminder_fired(rid) is False


def test_a_daily_reminder_comes_back_tomorrow(db, spawned, monkeypatch):
    db.create_reminder("standup", in_(-1), repeat_rule="daily")
    m = reminders.ReminderManager(db, {"notification_method": "none"})
    rung = []
    monkeypatch.setattr(m, "fire_reminder", lambda r: rung.append(r["text"]))
    m.run_once()
    m.run_once()
    assert rung == ["standup"] and abs(alarm.next_due() - in_(86400).timestamp()) < 3


def test_a_reminder_saved_while_the_alarm_cant_be_set_is_not_lost(db, spawned, monkeypatch):
    monkeypatch.setattr(alarm.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert db.create_reminder("still saved", in_(60))
    assert len(db.get_all_reminders()) == 1


# ── alerts that don't need a collector ────────────────────────────────────
def test_disk_alert_works_with_no_prometheus(db, monkeypatch):
    rule = {"type": "metric_threshold", "metric": "host_disk", "operator": ">", "threshold": 1}
    assert automation.AutomationEngine([], db=db)._check_condition(rule)
    assert not automation.AutomationEngine([], db=db)._check_condition({**rule, "threshold": 100})


def test_measured_numbers_are_sensible():
    assert 0 < automation.measure("host_disk") < 100
    assert 0 <= automation.measure("host_cpu") <= 100
    assert automation.measure("made_up") is None


def test_a_fresh_collector_reading_is_preferred(db, monkeypatch):
    monkeypatch.setattr(automation, "measure", lambda m: pytest.fail("a stored reading exists"))
    monkeypatch.setattr(db, "get_context", lambda **k: [{"key": "host_cpu", "value": {"usage_pct": 95}}])
    rule = {"type": "metric_threshold", "metric": "host_cpu", "operator": ">", "threshold": 80}
    assert automation.AutomationEngine([], db=db)._check_condition(rule)


# ── the control panel ─────────────────────────────────────────────────────
@pytest.fixture
def laptop(monkeypatch):
    monkeypatch.setattr(power, "battery_state", lambda: PLUGGED)


def plist():
    with open(scheduler.plist_path("tick"), "rb") as f:
        return plistlib.load(f)


@pytest.mark.skipif(os.name == "nt", reason="launchd")
def test_on_means_one_job_off_means_none(laptop):
    assert workers.set_enabled("reminders", True)["on"]
    assert workers.set_enabled("collector.git", True)["on"]
    jobs = os.listdir(os.path.dirname(scheduler.plist_path("tick")))
    assert jobs == ["com.cortex.tick.plist"]
    job = plist()
    assert job["StartInterval"] == 600 and job["RunAtLoad"] and job["AbandonProcessGroup"]
    assert job["ProgramArguments"][-2:][1] == "tick" and job["ProcessType"] == "Background"
    workers.set_enabled("reminders", False)
    assert workers.is_on("collector.git") and not workers.is_on("reminders")
    workers.set_enabled("collector.git", False)
    assert not scheduler.installed("tick")


def test_a_server_ticks_every_minute(monkeypatch):
    monkeypatch.setattr(power, "battery_state", lambda: DESKTOP)
    workers.set_enabled("reminders", True)
    assert plist()["StartInterval"] == 60


def test_saying_its_a_server_changes_the_job(laptop):
    workers.set_enabled("reminders", True)
    assert workers.set_machine("server")["ok"] and plist()["StartInterval"] == 60
    assert workers.set_machine("auto")["ok"] and plist()["StartInterval"] == 600
    assert not workers.set_machine("toaster")["ok"]


def test_the_panel_doesnt_claim_what_isnt_true(laptop, tmp_path):
    (tmp_path / ".launchd-refuses").touch()
    r = workers.set_enabled("reminders", True)
    assert not r["ok"] and not r["on"] and "wouldn't accept" in r["error"]


def test_a_job_removed_behind_our_back_shows_as_off_and_comes_back(laptop):
    workers.set_enabled("reminders", True)
    scheduler.remove("tick")
    assert not workers.is_on("reminders")
    assert not {r["key"]: r for r in workers.list_workers()}["reminders"]["on"]
    workers.ensure()
    assert workers.is_on("reminders")


def test_switching_on_an_ai_job_doesnt_start_it(laptop):
    workers.set_enabled("autonomy", True)
    workers.set_enabled("report-morning", True)
    last = tick.load_state()["last"]
    assert time.time() - last["autonomy"] < 5 and time.time() - last["report-morning"] < 5


def test_rows_say_what_will_happen(laptop):
    rows = {r["key"]: r for r in workers.list_workers()}
    assert rows["reminders"]["cadence"] == "exactly on time"
    assert rows["automation"]["cadence"] == "every 10 min, every 30 min on battery"
    assert rows["collector.git"]["cadence"] == "every 30 min, plugged in only"
    assert rows["report-morning"]["cadence"] == "daily 09:00, plugged in only"
    ss.update_section("resources", {"ai_jobs_only_on_ac": False})
    assert {r["key"]: r for r in workers.list_workers()}["report-morning"]["cadence"] == "daily 09:00"
    ss.update_section("resources", {"machine": "server"})
    assert {r["key"]: r for r in workers.list_workers()}["collector.process"]["cadence"] == "every 3 min"


def test_last_run_and_failures_show_up(laptop):
    tick.save_state({"last": {"collector.git": 123.0}, "exit": {"collector.git": 2}})
    row = {r["key"]: r for r in workers.list_workers()}["collector.git"]
    assert row["last_run"] == 123.0 and row["last_exit"] == 2


def test_the_note_says_how_its_powered(monkeypatch):
    monkeypatch.setattr(power, "battery_state", lambda: BATTERY)
    assert workers.power_info()["note"].startswith("On battery: only reminders and rules")
    monkeypatch.setattr(power, "battery_state", lambda: DESKTOP)
    assert workers.power_info()["kind"] == "server"


def test_an_old_job_stays_quiet_once_switched_off():
    assert workers.is_allowed("reminders")[0] is True     # not moved over yet
    ss.save({"workers_on": ["automation"]})
    assert workers.is_allowed("automation")[0] is True
    assert workers.is_allowed("reminders")[0] is False
    assert workers.is_allowed("report", "Morning Briefing")[0] is False


# ── moving off the old jobs ───────────────────────────────────────────────
def old_jobs(on):
    for w in workers.BUILTIN:
        path = scheduler.write_plist(w["key"], ["x"], {"type": "interval", "seconds": 60})
        if w["key"] in on:
            subprocess.run(["launchctl", "load", path])


def test_what_was_on_stays_on_and_the_old_jobs_go(laptop):
    old_jobs({"reminders", "collector.git"})
    rows = {r["key"]: r for r in workers.list_workers()}
    assert rows["reminders"]["on"] and rows["collector.git"]["on"] and not rows["autonomy"]["on"]
    assert os.listdir(os.path.dirname(scheduler.plist_path("tick"))) == ["com.cortex.tick.plist"]


def test_all_off_leaves_no_job_at_all_and_no_password_on_disk(laptop):
    old_jobs(set())
    workers.ensure()
    assert os.listdir(os.path.dirname(scheduler.plist_path("tick"))) == []
    assert ss.load()["workers_on"] == []


def test_a_fresh_install_has_nothing_to_move():
    workers.ensure()
    assert not scheduler.installed("tick") and workers.enabled_keys() == set()


def test_moving_happens_once(laptop):
    old_jobs({"reminders"})
    workers.ensure()
    workers.set_enabled("reminders", False)
    old_jobs({"reminders"})     # would only be turned back on by a second migration
    workers.ensure()
    assert workers.enabled_keys() == set()


# ── power readings on every OS ────────────────────────────────────────────
def supply(root, name, **files):
    d = root / name
    d.mkdir()
    for k, v in files.items():
        (d / k).write_text(v + "\n")


def linux(monkeypatch, tmp_path):
    root = tmp_path / "ps"
    root.mkdir()
    monkeypatch.setattr(power, "_SYS_POWER", str(root))
    return root


def test_linux_laptop_on_battery(monkeypatch, tmp_path):
    root = linux(monkeypatch, tmp_path)
    supply(root, "AC", type="Mains", online="0")
    supply(root, "BAT0", type="Battery", status="Discharging", capacity="41")
    assert power._linux_battery() == {"on_ac": False, "percent": 41, "charging": False, "has_battery": True}


def test_linux_laptop_full_and_plugged_in(monkeypatch, tmp_path):
    supply(linux(monkeypatch, tmp_path), "BAT0", type="Battery", status="Full", capacity="100")
    got = power._linux_battery()
    assert got["on_ac"] and got["has_battery"] and got["percent"] == 100


def test_a_wireless_mouse_doesnt_make_it_a_laptop(monkeypatch, tmp_path):
    supply(linux(monkeypatch, tmp_path), "hidpp_battery_0", type="Battery", scope="Device", status="Discharging", capacity="50")
    assert power._linux_battery() == power._NO_BATTERY


def test_linux_with_no_power_folder_is_a_server(monkeypatch, tmp_path):
    monkeypatch.setattr(power, "_SYS_POWER", str(tmp_path / "missing"))
    assert power._linux_battery() == power._NO_BATTERY


@pytest.mark.parametrize("ac,flag,pct,want", [
    (0, 0, 55, {"on_ac": False, "percent": 55, "charging": False, "has_battery": True}),
    (1, 8, 80, {"on_ac": True, "percent": 80, "charging": True, "has_battery": True}),
    (1, 128, 255, power._NO_BATTERY),
    (255, 255, 255, power._NO_BATTERY),
])
def test_windows_readings(ac, flag, pct, want):
    assert power._windows_reading(ac, flag, pct) == want


def test_macbook_and_mac_mini(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power, "_run", lambda c, timeout=4: "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t63%; discharging; 4:12 remaining present: true")
    got = power.battery_state()
    assert not got["on_ac"] and got["has_battery"] and got["percent"] == 63
    monkeypatch.setattr(power, "_run", lambda c, timeout=4: "Now drawing from 'AC Power'\n")
    got = power.battery_state()
    assert got["on_ac"] and not got["has_battery"]


def test_waiting_loops_slow_down_on_battery(monkeypatch):
    calls = []
    monkeypatch.setitem(power._paced, "checked", None)
    monkeypatch.setattr(power, "battery_state", lambda: calls.append(1) or BATTERY)
    assert power.pace(4) == 30 and power.pace(5) == 30 and power.pace(60) == 60
    assert len(calls) == 1      # the check mustn't become the wake-up
    monkeypatch.setitem(power._paced, "checked", None)
    monkeypatch.setattr(power, "battery_state", lambda: PLUGGED)
    assert power.pace(4) == 4
