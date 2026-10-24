# legacy/ — archived code

These files are **not used by the running app**. They're the old always-on Flask
daemon (`cortex.py`, retired in favour of the scheduled `run_once` model + the
in-process `cortex_core`) and the modules only it imported:

- `cortex.py` — old :7800 daemon
- `autonomy.py` — single-shot autonomy (superseded by `cortex/core/autonomy_session.py`)
- `appflowy.py`, `clipboard_watch.py`, `credential_prompt.py`, `devops_agent.py`,
  `multi_machine.py`, `multi_scope.py`, `vision.py` — daemon-only features
- `migrate_to_sqlite.py` — one-time Postgres→SQLite migration (already run)
- `test_connection.py` — Postgres-era connection/schema tester
- `com.cortex.daemon.plist` — the retired launchd job

Their imports reference the old flat module layout and will NOT resolve against the
`cortex/` package. Kept for reference only; safe to delete.
