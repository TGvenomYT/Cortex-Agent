"""
Project Collector — runs every 30 minutes.
Captures project type, dependencies, structure, TODO count,
and any custom commands registered for each project.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context, run_cmd, get_tracked_projects

logger = logging.getLogger("collector.project")


def detect_project_type(path: str) -> str:
    markers = {
        "package.json": "node",
        "requirements.txt": "python",
        "pyproject.toml": "python",
        "Cargo.toml": "rust",
        "go.mod": "go",
        "pom.xml": "java",
        "build.gradle": "java",
        "Gemfile": "ruby",
        "composer.json": "php",
        "pubspec.yaml": "flutter",
        "Makefile": "make",
        "Dockerfile": "docker",
    }
    for filename, ptype in markers.items():
        if os.path.exists(os.path.join(path, filename)):
            return ptype
    return "unknown"


def collect_dependencies(path: str, ptype: str) -> dict:
    deps = {"type": ptype, "count": 0, "names": []}

    if ptype == "node":
        pkg_path = os.path.join(path, "package.json")
        try:
            with open(pkg_path) as f:
                pkg = json.load(f)
            all_deps = list(pkg.get("dependencies", {}).keys()) + list(pkg.get("devDependencies", {}).keys())
            deps["count"] = len(all_deps)
            deps["names"] = all_deps[:30]
            deps["scripts"] = list(pkg.get("scripts", {}).keys())
            deps["version"] = pkg.get("version", "")
            deps["name"] = pkg.get("name", "")
        except Exception:
            pass

    elif ptype == "python":
        req_path = os.path.join(path, "requirements.txt")
        try:
            with open(req_path) as f:
                lines = [l.strip().split(">=")[0].split("==")[0].split("[")[0]
                         for l in f if l.strip() and not l.startswith("#")]
            deps["count"] = len(lines)
            deps["names"] = lines[:30]
        except Exception:
            pass

    elif ptype == "rust":
        out, _, _ = run_cmd("cargo metadata --no-deps --format-version 1 2>/dev/null", cwd=path)
        try:
            meta = json.loads(out)
            pkgs = meta.get("packages", [])
            deps["count"] = len(pkgs)
            deps["names"] = [p["name"] for p in pkgs[:30]]
        except Exception:
            pass

    elif ptype == "go":
        out, _, _ = run_cmd("go list -m all 2>/dev/null", cwd=path)
        mods = [l.split()[0] for l in out.splitlines() if l.strip()]
        deps["count"] = len(mods)
        deps["names"] = mods[:30]

    return deps


def collect_structure(path: str) -> dict:
    """Top-level directory structure + file counts."""
    try:
        entries = os.listdir(path)
    except Exception:
        return {}

    dirs = []
    files = []
    for e in sorted(entries):
        if e.startswith("."):
            continue
        full = os.path.join(path, e)
        if os.path.isdir(full):
            dirs.append(e)
        else:
            files.append(e)

    # Count total files (non-recursive, excluding node_modules/.git)
    total_files = 0
    for root, subdirs, filenames in os.walk(path):
        subdirs[:] = [d for d in subdirs if d not in {"node_modules", ".git", "__pycache__", ".venv", "dist", "build"}]
        total_files += len(filenames)
        if total_files > 10000:
            break

    return {
        "top_dirs": dirs[:20],
        "top_files": files[:20],
        "total_files_estimate": total_files,
    }


def collect_todos(path: str) -> dict:
    """Count TODO/FIXME/HACK comments across the codebase."""
    out, _, _ = run_cmd(
        "grep -r --include='*.py' --include='*.js' --include='*.ts' "
        "--include='*.go' --include='*.rs' --include='*.rb' "
        "-E '(TODO|FIXME|HACK|XXX):?' . 2>/dev/null | head -30",
        cwd=path,
        timeout=15
    )
    todos = []
    for line in out.splitlines():
        # format: ./file.py:12:  # TODO: fix this
        parts = line.split(":", 2)
        if len(parts) >= 3:
            todos.append({
                "file": parts[0].lstrip("./"),
                "line": parts[1],
                "text": parts[2].strip()[:100],
            })
    return {"count": len(todos), "items": todos[:10]}


def main():
    config = load_config()
    conn = get_db_conn(config)

    projects = get_tracked_projects(conn)
    if not projects:
        logger.warning("No tracked projects. Add via POST /projects/track")
        return

    for p in projects:
        path = p["path"]
        name = p["name"]
        if not os.path.exists(path):
            logger.warning(f"{name}: path does not exist ({path})")
            continue

        try:
            ptype = detect_project_type(path)
            deps = collect_dependencies(path, ptype)
            structure = collect_structure(path)
            todos = collect_todos(path)

            data = {
                "project": name,
                "path": path,
                "type": ptype,
                "dependencies": deps,
                "structure": structure,
                "todos": todos,
                "custom_commands": p.get("custom_commands") or {},
            }

            upsert_context(conn, "project", "info", data, project_path=path, expires_minutes=60)
            logger.info(f"{name}: type={ptype}, deps={deps['count']}, todos={todos['count']}")

        except Exception as e:
            logger.error(f"Failed project collection for {name}: {e}")

    conn.close()
    logger.info(f"Project collection done for {len(projects)} projects")


if __name__ == "__main__":
    main()
