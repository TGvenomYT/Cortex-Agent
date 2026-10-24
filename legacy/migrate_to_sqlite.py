#!/usr/bin/env python3
"""
migrate_to_sqlite.py — one-time copy of the remote Postgres data into the local
SQLite store, so going local-first preserves everything the agent already knows
(memory, priorities, goals, autonomy history, experiences).

Run once while Tailscale is up:  python3 migrate_to_sqlite.py
Idempotent: rows are inserted by primary key with INSERT OR IGNORE, so running
it again won't duplicate.
"""
from __future__ import annotations

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("migrate")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from cortex_bootstrap import load_config, load_env_file

# context_store is deliberately NOT migrated: it is ephemeral (collectors
# regenerate it every minute, rows expire), and on the server it had ballooned
# to ~149k rows. Migrating it is pointless and slow. Everything below is the
# agent's actual durable knowledge/state.
TABLES = [
    "memory", "priorities", "goals", "autonomy_log", "experiences",
    "reminders", "tracked_projects", "devops_plans", "conversations",
    "command_log",
]


def main() -> int:
    load_env_file()
    config = load_config()
    dbc = config["database"]

    # Source: Postgres
    import psycopg2
    from psycopg2 import extras
    pw = dbc.get("password") or os.environ.get("CORTEX_DB_PASSWORD", "")
    if not pw:
        log.error("No Postgres password available (set CORTEX_DB_PASSWORD or ~/.cortex/cortex.env).")
        return 1
    try:
        pg = psycopg2.connect(host=dbc["host"], port=dbc["port"], dbname=dbc["name"],
                              user=dbc["user"], password=pw, connect_timeout=10)
    except Exception as e:
        log.error(f"Cannot reach Postgres (is Tailscale up?): {e}")
        return 1

    # Destination: SQLite
    import sqlite_backend
    sqlite_path = os.path.expanduser(dbc.get("sqlite_path", "~/.cortex/cortex.db"))
    slite = sqlite_backend.open_db(sqlite_path)

    total = 0
    for table in TABLES:
        try:
            with pg.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM {table}")
                rows = cur.fetchall()
        except Exception as e:
            log.info(f"  skip {table}: {e}")
            pg.rollback()
            continue
        if not rows:
            log.info(f"  {table}: 0 rows")
            continue

        # Only migrate columns that exist in the local schema.
        local_cols = {r[1] for r in slite.execute(f"PRAGMA table_info({table})").fetchall()}
        copied = 0
        for row in rows:
            d = {k: v for k, v in dict(row).items() if k in local_cols}
            for k, v in list(d.items()):
                if isinstance(v, (dict, list)):
                    d[k] = json.dumps(v)          # JSONB -> text
                elif isinstance(v, bool):
                    d[k] = 1 if v else 0          # bool -> int
            cols = ", ".join(d.keys())
            marks = ", ".join(["?"] * len(d))
            try:
                slite.execute(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({marks})",
                              tuple(d.values()))
                copied += 1
            except Exception as e:
                log.info(f"    row skip in {table}: {e}")
        slite.commit()
        total += copied
        log.info(f"  {table}: {copied}/{len(rows)} rows")

    slite.close()
    pg.close()
    log.info(f"Done. {total} rows migrated into {sqlite_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
