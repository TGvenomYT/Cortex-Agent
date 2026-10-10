"""
GitHub Collector — Pulls repo status, PRs, issues from GitHub via gh CLI.
Runs every 5 minutes.
"""

import os
import sys
import json
import subprocess
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context

logger = logging.getLogger("collector.github")


def _gh(args: str, timeout: int = 15) -> str:
    """Run a gh CLI command."""
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    try:
        result = subprocess.run(
            f"gh {args}",
            shell=True, capture_output=True, text=True, timeout=timeout, env=env
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        logger.error(f"gh command failed: {e}")
        return ""


def collect_repos(conn):
    """List all repos with recent activity."""
    output = _gh("repo list --limit 20 --json name,updatedAt,visibility,description,primaryLanguage")
    if not output:
        return

    repos = json.loads(output)
    repo_summary = []
    for r in repos:
        repo_summary.append({
            "name": r.get("name", ""),
            "visibility": r.get("visibility", "").lower(),
            "language": r.get("primaryLanguage", {}).get("name", "") if r.get("primaryLanguage") else "",
            "description": (r.get("description") or "")[:100],
            "updated": r.get("updatedAt", "")[:10],
        })

    upsert_context(conn, "github", "repos", {
        "count": len(repo_summary),
        "repos": repo_summary,
    }, expires_minutes=15)
    logger.info(f"GitHub: {len(repo_summary)} repos")


def collect_notifications(conn):
    """Unread GitHub notifications."""
    output = _gh("api notifications --jq '.[].subject | {title, type: .type, url: .url}' 2>/dev/null")
    if not output:
        upsert_context(conn, "github", "notifications", {"count": 0, "items": []}, expires_minutes=10)
        return

    notifications = []
    for line in output.splitlines():
        try:
            n = json.loads(line)
            notifications.append(n)
        except Exception:
            pass

    upsert_context(conn, "github", "notifications", {
        "count": len(notifications),
        "items": notifications[:10],
    }, expires_minutes=10)
    logger.info(f"GitHub: {len(notifications)} unread notifications")


def collect_open_prs(conn):
    """Open PRs across repos."""
    output = _gh("search prs --author=@me --state=open --json title,repository,url,updatedAt --limit 10")
    if not output:
        upsert_context(conn, "github", "open_prs", {"count": 0, "prs": []}, expires_minutes=15)
        return

    prs = json.loads(output)
    pr_summary = []
    for pr in prs:
        pr_summary.append({
            "title": pr.get("title", ""),
            "repo": pr.get("repository", {}).get("name", ""),
            "updated": pr.get("updatedAt", "")[:10],
        })

    upsert_context(conn, "github", "open_prs", {
        "count": len(pr_summary),
        "prs": pr_summary,
    }, expires_minutes=15)
    logger.info(f"GitHub: {len(pr_summary)} open PRs")


def collect_recent_activity(conn):
    """Recent commits/events."""
    output = _gh("api users/{owner}/events --jq '.[0:5] | .[] | {type, repo: .repo.name, created: .created_at}' 2>/dev/null")
    # This might not work without knowing the username, fallback to repo-specific
    if not output:
        # Try getting recent commits from top repos
        repos_out = _gh("repo list --limit 3 --json name")
        if repos_out:
            repos = json.loads(repos_out)
            activity = []
            for r in repos[:3]:
                name = r.get("name", "")
                commits = _gh(f"api repos/{{owner}}/{name}/commits --jq '.[0:3] | .[].commit | {{message: .message, date: .author.date}}' 2>/dev/null")
                if commits:
                    for line in commits.splitlines():
                        try:
                            c = json.loads(line)
                            c["repo"] = name
                            activity.append(c)
                        except Exception:
                            pass

            upsert_context(conn, "github", "recent_activity", {
                "commits": activity[:10],
            }, expires_minutes=15)
            logger.info(f"GitHub: {len(activity)} recent commits collected")
            return

    upsert_context(conn, "github", "recent_activity", {"commits": []}, expires_minutes=15)


def main():
    config = load_config()
    conn = get_db_conn(config)

    collect_repos(conn)
    collect_notifications(conn)
    collect_open_prs(conn)
    collect_recent_activity(conn)

    conn.close()
    logger.info("GitHub collection done")


if __name__ == "__main__":
    main()
