"""
Git Collector — runs every 2 minutes.
Captures branch, status, last commit, ahead/behind, stash count
for every tracked project that has a .git directory.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context, run_cmd, get_tracked_projects

logger = logging.getLogger("collector.git")


def collect_git(conn, project_path: str, project_name: str):
    git_dir = os.path.join(project_path, ".git")
    if not os.path.exists(git_dir):
        logger.debug(f"{project_name}: no .git, skipping")
        return

    # Current branch
    branch, _, _ = run_cmd("git rev-parse --abbrev-ref HEAD", cwd=project_path)

    # Status — uncommitted files
    status_out, _, _ = run_cmd("git status --porcelain", cwd=project_path)
    uncommitted = [l for l in status_out.splitlines() if l.strip()]

    # Last commit
    last_commit, _, _ = run_cmd(
        'git log -1 --format="%h|%s|%ar|%an"', cwd=project_path
    )
    commit_parts = last_commit.split("|") if last_commit else []
    commit = {
        "hash": commit_parts[0] if len(commit_parts) > 0 else "",
        "message": commit_parts[1] if len(commit_parts) > 1 else "",
        "when": commit_parts[2] if len(commit_parts) > 2 else "",
        "author": commit_parts[3] if len(commit_parts) > 3 else "",
    }

    # Ahead/behind remote
    ahead_behind, _, rc = run_cmd(
        "git rev-list --left-right --count HEAD...@{upstream}", cwd=project_path
    )
    if rc == 0 and ahead_behind:
        parts = ahead_behind.split()
        ahead = int(parts[0]) if len(parts) > 0 else 0
        behind = int(parts[1]) if len(parts) > 1 else 0
    else:
        ahead, behind = 0, 0

    # Remote URL
    remote_url, _, _ = run_cmd("git remote get-url origin", cwd=project_path)

    # Stash count
    stash_out, _, _ = run_cmd("git stash list", cwd=project_path)
    stash_count = len(stash_out.splitlines()) if stash_out else 0

    # Recent branches
    branches_out, _, _ = run_cmd(
        "git branch --sort=-committerdate --format='%(refname:short)' | head -5",
        cwd=project_path
    )
    recent_branches = [b.strip() for b in branches_out.splitlines() if b.strip()]

    data = {
        "project": project_name,
        "path": project_path,
        "branch": branch,
        "uncommitted_count": len(uncommitted),
        "uncommitted_files": uncommitted[:10],  # cap at 10
        "last_commit": commit,
        "ahead": ahead,
        "behind": behind,
        "stash_count": stash_count,
        "recent_branches": recent_branches,
        "remote_url": remote_url,
        "clean": len(uncommitted) == 0,
    }

    upsert_context(conn, "git", "status", data, project_path=project_path, expires_minutes=5)
    logger.info(f"{project_name}: branch={branch}, uncommitted={len(uncommitted)}, ahead={ahead}, behind={behind}")


def main():
    config = load_config()
    conn = get_db_conn(config)

    projects = get_tracked_projects(conn)
    if not projects:
        logger.warning("No tracked projects in DB. Add them via POST /projects/track")
        return

    for p in projects:
        try:
            collect_git(conn, p["path"], p["name"])
        except Exception as e:
            logger.error(f"Failed git collection for {p['name']}: {e}")

    conn.close()
    logger.info(f"Git collection done for {len(projects)} projects")


if __name__ == "__main__":
    main()
