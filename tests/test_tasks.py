import pytest

from cortex.core import tasks


@pytest.mark.parametrize("text,expect", [
    ("every 30 min", {"type": "interval", "seconds": 1800}),
    ("every 90 minutes", {"type": "interval", "seconds": 5400}),
    ("every 2 hours", {"type": "interval", "seconds": 7200}),
    ("every 2 days", {"type": "interval", "seconds": 172800}),
    ("hourly", {"type": "interval", "seconds": 3600}),
    ("daily 08:30", {"type": "daily", "hour": 8, "minute": 30}),
    ("at 5pm", {"type": "daily", "hour": 17, "minute": 0}),
    ("at 12am", {"type": "daily", "hour": 0, "minute": 0}),
    ("12:30 pm", {"type": "daily", "hour": 12, "minute": 30}),
    ("every morning", {"type": "daily", "hour": 8, "minute": 0}),
    ("every evening", {"type": "daily", "hour": 20, "minute": 0}),
    ("daily", {"type": "daily", "hour": 9, "minute": 0}),
    ("every friday", {"type": "weekly", "weekday": 5, "hour": 9, "minute": 0}),
    ("every monday at 9am", {"type": "weekly", "weekday": 1, "hour": 9, "minute": 0}),
    ("weekly", {"type": "weekly", "weekday": 1, "hour": 9, "minute": 0}),
])
def test_schedules_mean_what_they_say(text, expect):
    assert tasks.parse_schedule(text)[0] == expect


@pytest.mark.parametrize("text", ["every 1 min", "every 30 seconds", "every 5 minutes"])
def test_ai_tasks_have_a_floor(text):
    got, human = tasks.parse_schedule(text)
    assert got == {"type": "interval", "seconds": tasks.MIN_TASK_SECONDS}
    assert "5 minutes" in human


@pytest.mark.parametrize("text", ["25:99", "at 13pm", "whenever", "", "in 10 minutes", "tomorrow at 9", "once"])
def test_unreadable_schedule_is_refused_not_guessed(text):
    with pytest.raises(ValueError):
        tasks.parse_schedule(text)


def test_failed_add_stores_nothing(monkeypatch):
    monkeypatch.setattr(tasks, "_install", lambda t: None)
    with pytest.raises(ValueError):
        tasks.add_task("d", "p", "whenever")
    assert tasks.list_tasks() == []


def test_failures_count_up_and_clear(monkeypatch):
    monkeypatch.setattr(tasks, "_install", lambda t: None)
    t = tasks.add_task("d", "p", "hourly")
    assert tasks.record_failure(t["id"]) == 1
    assert tasks.record_failure(t["id"]) == 2
    tasks.clear_failures(t["id"])
    assert tasks.get_task(t["id"])["failures"] == 0
