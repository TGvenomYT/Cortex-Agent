"""Scheduled tasks, sync, retrieval: what runs when nobody's watching."""
import argparse
import os

import pytest

from cortex.core import tasks as tasks_mod


@pytest.fixture
def runner(monkeypatch):
    """cmd_task with a stub agent. .answer is what it replies; .sent is what reached the user."""
    import cortex.core.cortex_core as cc
    import cortex.server.run_once as ro
    monkeypatch.setattr(tasks_mod, "_install", lambda t: None)
    monkeypatch.setattr(tasks_mod, "_uninstall", lambda t: None)
    monkeypatch.setattr(ro.power, "should_run", lambda *a, **k: (True, "ok"))

    class R:
        answer, error, sent, nudges = "", False, [], []
        prompts, sources = [], []

    def ask_full(prompt, session_id, source):
        R.prompts.append(prompt)
        R.sources.append(source)
        return {"speak": R.answer, "error": R.error}

    monkeypatch.setattr(cc, "ask_full", ask_full)
    monkeypatch.setattr(cc, "_CORE", type("C", (), {"db": type("D", (), {
        "add_nudge": lambda self, t, b, kind, source: R.nudges.append(b)})()})())
    monkeypatch.setattr(ro, "_notify", lambda t, m: R.sent.append(m))
    R.sent, R.nudges, R.prompts, R.sources = [], [], [], []
    R.task = tasks_mod.add_task("watch it", "check the thing", "hourly")
    R.run = lambda: ro.cmd_task(argparse.Namespace(name=R.task["id"], force=False, dry_run=False))
    return R


def test_a_run_is_a_background_source_and_knows_the_rules(runner):
    runner.answer = "something happened"
    runner.run()
    assert runner.sources == ["task"] and "NOTHING_TO_REPORT" in runner.prompts[0]


def test_something_to_say_is_delivered(runner):
    runner.answer = "disk is full"
    runner.run()
    assert runner.sent == ["disk is full"] and runner.nudges == ["disk is full"]


def test_nothing_to_report_is_silent(runner):
    runner.answer = "NOTHING_TO_REPORT"
    runner.run()
    assert runner.sent == [] and runner.nudges == []
    assert tasks_mod.get_task(runner.task["id"])


def test_a_finished_job_says_so_once_and_removes_itself(runner):
    runner.answer = "TASK_COMPLETE: the emulator is gone"
    runner.run()
    assert runner.sent == ["the emulator is gone"]
    assert tasks_mod.get_task(runner.task["id"]) is None


def test_no_model_stays_silent_then_pauses_with_one_message(runner):
    runner.answer, runner.error = "I can't reach a model", True
    runner.run()
    runner.run()
    assert runner.sent == []
    runner.run()
    assert len(runner.sent) == 1 and "paused" in runner.sent[0]
    assert tasks_mod.get_task(runner.task["id"])["enabled"] is False


def test_recovering_resets_the_count(runner):
    runner.answer, runner.error = "x", True
    runner.run()
    runner.answer, runner.error = "fine", False
    runner.run()
    assert tasks_mod.get_task(runner.task["id"])["failures"] == 0


def test_a_paused_task_is_refused_at_the_door(runner):
    import cortex.core.workers as workers
    tasks_mod.set_task_enabled(runner.task["id"], False)
    assert workers.is_allowed("task", runner.task["id"])[0] is False


def test_profile_is_not_off_while_a_task_runs(monkeypatch):
    import cortex.core.workers as workers
    rows = [{"key": "reminders", "group": "essentials", "on": False},
            {"key": "task.a", "group": "tasks", "on": True}]
    assert workers.current_profile(rows) == "custom"
    rows[1]["on"] = False
    assert workers.current_profile(rows) == "off"


class FakePG:
    """Enough of a Postgres connection to show what a real transaction would keep."""
    def __init__(self):
        self.buf, self.done, self.saved = [], [], []
        pg = self

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def execute(self, sql, params=None):
                if sql.startswith("SAVEPOINT"):
                    pg.saved = list(pg.buf)
                elif sql.startswith("ROLLBACK TO"):
                    pg.buf = list(pg.saved)
                elif sql.startswith("RELEASE"):
                    pass
                elif params[0] == 3:
                    raise RuntimeError("bad row")
                else:
                    pg.buf.append(params[0])
        self.Cur = Cur

    def cursor(self):
        return self.Cur()

    def commit(self):
        self.done += self.buf
        self.buf = []


def test_a_bad_row_loses_nothing_else_and_is_retried():
    from cortex.storage import sync
    pg = FakePG()
    pushed, failed = sync._push_table(None, pg, "conversations", ["id"], [{"id": i} for i in (1, 2, 3, 4, 5)], False)
    assert pg.done == [1, 2, 4, 5] and pushed == 4 and failed == 3


def test_retrieval_works_with_no_embedding_model(make_core, tmp_path):
    from cortex.storage import rag
    core = make_core()
    doc = tmp_path / "notes.txt"
    doc.write_text("The kestrel project ships on the ninth of March.")
    assert rag.ingest(core.db, str(doc))[1] == 1
    hits = rag.retrieve(core.db, "when does kestrel ship", k=3)
    assert hits and "ninth of March" in hits[0]["chunk"] and hits[0]["similarity"] >= 0.4
    assert rag.retrieve(core.db, "weather in Paris", k=3) == []


def test_uploading_a_file_again_replaces_it(make_core, tmp_path):
    from cortex.storage import rag
    core = make_core()
    doc = tmp_path / "notes.txt"
    doc.write_text("first version")
    rag.ingest(core.db, str(doc))
    rag.ingest(core.db, str(doc))
    assert len(core.db.search_documents_keyword("first version")) == 1


def test_reading_a_pdf_with_no_pdftotext(tmp_path, monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    from cortex.storage import rag
    w = pypdf.PdfWriter()
    w.add_blank_page(72, 72)
    path = tmp_path / "a.pdf"
    with open(path, "wb") as f:
        w.write(f)
    monkeypatch.setattr(rag.shutil, "which", lambda name: None)
    assert rag._pdf(str(path)) == ""


def test_calendar_miss_is_remembered_across_processes(monkeypatch, tmp_path):
    from cortex.integrations import calendar_integration as cal
    monkeypatch.setattr(cal, "_CAL_FILE", str(tmp_path / "cal.json"))
    monkeypatch.setattr(cal, "_CAL_CACHE", {})
    monkeypatch.setattr(cal.subprocess if hasattr(cal, "subprocess") else None, "run",
                        lambda *a, **k: (_ for _ in ()).throw(cal.subprocess.TimeoutExpired("x", 1)))
    cal._run_calendar_script("s")
    assert os.path.exists(cal._CAL_FILE)
    cal._CAL_CACHE.clear()      # a new process
    monkeypatch.setattr(cal.subprocess, "run", lambda *a, **k: pytest.fail("should not have asked Calendar again"))
    assert cal._run_calendar_script("s") == []


def test_project_detection(tmp_path):
    from cortex.core import projects
    (tmp_path / "package.json").write_text('{"scripts":{"test":"x","dev":"y"}}')
    assert projects.command_for(str(tmp_path), "test") == "npm run test"
    assert projects.command_for(str(tmp_path), "build") is None
    (tmp_path / "yarn.lock").write_text("")
    assert projects.command_for(str(tmp_path), "install") == "yarn install"


def test_run_project_runs_the_step_not_the_word(make_core, tmp_path):
    from cortex.core import projects
    (tmp_path / "Makefile").write_text("test:\n\t@echo ran-the-tests\n")
    out = projects.run(str(tmp_path), "test", make_core().executor)
    assert "ran-the-tests" in out and "exit 0" in out
