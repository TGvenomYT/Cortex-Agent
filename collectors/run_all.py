"""
run_all.py — Manual test runner for all collectors.
Run this to verify all collectors work and check what's stored in DB.

Usage:
  CORTEX_DB_PASSWORD=... python3 collectors/run_all.py
  CORTEX_DB_PASSWORD=... python3 collectors/run_all.py --show-db
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn

import git_collector
import process_collector
import system_collector
import shell_collector
import project_collector


COLLECTORS = [
    ("system",  system_collector,  "Disk, RAM, CPU, uptime, network"),
    ("shell",   shell_collector,   "zsh history"),
    ("process", process_collector, "Docker, ports, services"),
    ("git",     git_collector,     "Git state for tracked projects"),
    ("project", project_collector, "Project type, deps, TODOs"),
]


def run_all():
    print("\n" + "═" * 60)
    print("  Cortex — Running all collectors")
    print("═" * 60 + "\n")

    results = []
    for name, module, description in COLLECTORS:
        print(f"  ▶ {name:<10} {description}")
        start = time.time()
        try:
            module.main()
            elapsed = int((time.time() - start) * 1000)
            print(f"  ✓ {name:<10} done in {elapsed}ms\n")
            results.append((name, True, elapsed, None))
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            print(f"  ✗ {name:<10} FAILED: {e}\n")
            results.append((name, False, elapsed, str(e)))

    # Summary
    print("═" * 60)
    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"  {passed}/{len(results)} collectors succeeded")
    print("═" * 60 + "\n")
    return results


def show_db_context():
    """Print what's currently stored in context_store."""
    config = load_config()
    conn = get_db_conn(config)

    import psycopg2.extras
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT category, key, project_path, collected_at,
                   expires_at, value
            FROM context_store
            ORDER BY category, collected_at DESC
        """)
        rows = cur.fetchall()

    conn.close()

    if not rows:
        print("  (no context in DB yet)")
        return

    print(f"\n  {'CATEGORY':<12} {'KEY':<25} {'PROJECT':<30} {'COLLECTED'}")
    print("  " + "─" * 90)
    for r in rows:
        project = (r["project_path"] or "")
        project = project.split("/")[-1] if project else "system"
        collected = str(r["collected_at"])[:19]
        print(f"  {r['category']:<12} {r['key']:<25} {project:<30} {collected}")

        # Show value summary
        val = r["value"]
        if isinstance(val, dict):
            # Print a one-line summary of the value
            summary_keys = list(val.keys())[:4]
            summary = ", ".join(f"{k}={str(val[k])[:20]}" for k in summary_keys)
            print(f"  {'':12} → {summary}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all Cortex collectors")
    parser.add_argument("--show-db", action="store_true", help="Show context_store contents after running")
    parser.add_argument("--db-only", action="store_true", help="Only show DB contents, don't run collectors")
    args = parser.parse_args()

    if not args.db_only:
        run_all()

    if args.show_db or args.db_only:
        print("\n  Context store contents:")
        print("  " + "═" * 90)
        show_db_context()
