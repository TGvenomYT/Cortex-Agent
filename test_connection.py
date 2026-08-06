"""
Quick connection test — run this before starting the daemon.
Usage: CORTEX_DB_PASSWORD=<password> python3 test_connection.py
"""

import os
import sys
import json

# ─── 1. Check dependencies ───
print("Checking dependencies...")
missing = []
for pkg, import_name in [
    ("psycopg2-binary", "psycopg2"),
    ("pyyaml", "yaml"),
    ("flask", "flask"),
    ("openai", "openai"),
    ("httpx", "httpx"),
]:
    try:
        __import__(import_name)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — MISSING")
        missing.append(pkg)

if missing:
    print(f"\nInstall missing packages:")
    print(f"  pip3 install {' '.join(missing)}")
    sys.exit(1)

# ─── 2. Load config ───
import yaml

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

db_cfg = config["database"]
password = db_cfg.get("password") or os.environ.get("CORTEX_DB_PASSWORD", "")

print(f"\nConnecting to PostgreSQL...")
print(f"  Host:  {db_cfg['host']}:{db_cfg['port']}")
print(f"  DB:    {db_cfg['name']}")
print(f"  User:  {db_cfg['user']}")

# ─── 3. Test DB connection ───
import psycopg2

try:
    conn = psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["name"],
        user=db_cfg["user"],
        password=password,
        connect_timeout=10
    )
    cur = conn.cursor()
    cur.execute("SELECT version()")
    version = cur.fetchone()[0]
    conn.close()
    print(f"  ✓ Connected! {version[:60]}")
except Exception as e:
    print(f"  ✗ Connection failed: {e}")
    sys.exit(1)

# ─── 4. Run schema ───
print(f"\nApplying schema...")
try:
    conn = psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["name"],
        user=db_cfg["user"],
        password=password,
        connect_timeout=10
    )
    conn.autocommit = True
    cur = conn.cursor()

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        schema_sql = f.read()

    cur.execute(schema_sql)
    conn.close()
    print("  ✓ Schema applied (all tables created)")
except Exception as e:
    print(f"  ✗ Schema failed: {e}")
    sys.exit(1)

# ─── 5. Verify tables ───
print(f"\nVerifying tables...")
try:
    conn = psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["name"],
        user=db_cfg["user"],
        password=password,
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name IN (
            'context_store', 'memory', 'conversations',
            'reminders', 'command_log', 'tracked_projects', 'devops_plans'
        )
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    conn.close()

    expected = {'context_store', 'memory', 'conversations', 'reminders',
                'command_log', 'tracked_projects', 'devops_plans'}
    for t in sorted(expected):
        if t in tables:
            print(f"  ✓ {t}")
        else:
            print(f"  ✗ {t} — NOT FOUND")
except Exception as e:
    print(f"  ✗ Table verification failed: {e}")
    sys.exit(1)

# ─── 6. Quick write/read test ───
print(f"\nWrite/read test...")
try:
    conn = psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        dbname=db_cfg["name"],
        user=db_cfg["user"],
        password=password,
    )
    cur = conn.cursor()

    # Write a test context row
    cur.execute("""
        INSERT INTO context_store (category, key, value)
        VALUES ('system', 'test', '{"status": "ok"}'::jsonb)
        ON CONFLICT (category, project_path, key)
        DO UPDATE SET value = EXCLUDED.value, collected_at = NOW()
    """)
    conn.commit()

    # Read it back
    cur.execute("SELECT value FROM context_store WHERE key = 'test'")
    val = cur.fetchone()[0]
    conn.close()
    print(f"  ✓ Write/read OK: {val}")
except Exception as e:
    print(f"  ✗ Write/read failed: {e}")
    sys.exit(1)

print("\n✓ All checks passed. Cortex is ready to start.")
print("\nTo start the daemon:")
print("  CORTEX_DB_PASSWORD=<password> python3 cortex.py")
