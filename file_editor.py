"""
File Editor — The agent's hands for reading and writing code.
Allows Cortex to read files, apply edits, create new files, and run projects.

This is what makes Cortex a real coding agent — not just a chatbot that tells you
what to do, but one that actually does it.
"""

import os
import re
import difflib
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.file_editor")

# Max file size we'll read into LLM context (200KB)
MAX_READ_SIZE = 200_000

# Common project runners
PROJECT_RUNNERS = {
    "package.json": {
        "detect": "package.json",
        "dev": "npm run dev",
        "start": "npm start",
        "build": "npm run build",
        "test": "npm test",
        "install": "npm install",
    },
    "Cargo.toml": {
        "detect": "Cargo.toml",
        "dev": "cargo run",
        "start": "cargo run --release",
        "build": "cargo build",
        "test": "cargo test",
    },
    "pyproject.toml": {
        "detect": "pyproject.toml",
        "dev": "python -m uvicorn main:app --reload",
        "start": "python main.py",
        "build": "python -m build",
        "test": "pytest",
        "install": "pip install -e .",
    },
    "requirements.txt": {
        "detect": "requirements.txt",
        "start": "python main.py",
        "test": "pytest",
        "install": "pip install -r requirements.txt",
    },
    "Makefile": {
        "detect": "Makefile",
        "dev": "make dev",
        "start": "make run",
        "build": "make build",
        "test": "make test",
    },
    "docker-compose.yml": {
        "detect": "docker-compose.yml",
        "dev": "docker compose up",
        "start": "docker compose up -d",
        "build": "docker compose build",
        "stop": "docker compose down",
    },
    "go.mod": {
        "detect": "go.mod",
        "dev": "go run .",
        "start": "go run .",
        "build": "go build",
        "test": "go test ./...",
    },
}


def read_file(filepath: str) -> dict:
    """
    Read a file's content.
    Returns: {success, filepath, content, lines, size, language, error}
    """
    filepath = os.path.expanduser(filepath)

    if not os.path.exists(filepath):
        return {"success": False, "filepath": filepath, "error": "File not found", "content": None}

    if not os.path.isfile(filepath):
        return {"success": False, "filepath": filepath, "error": "Not a file (maybe a directory?)", "content": None}

    size = os.path.getsize(filepath)
    if size > MAX_READ_SIZE:
        return {
            "success": False,
            "filepath": filepath,
            "error": f"File too large ({size // 1024}KB). Max is {MAX_READ_SIZE // 1024}KB.",
            "content": None,
        }

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        language = _detect_language(filepath)
        line_count = content.count("\n") + 1

        return {
            "success": True,
            "filepath": filepath,
            "content": content,
            "lines": line_count,
            "size": size,
            "language": language,
            "error": None,
        }
    except Exception as e:
        return {"success": False, "filepath": filepath, "error": str(e), "content": None}


def write_file(filepath: str, content: str, create_dirs: bool = True) -> dict:
    """
    Write content to a file (create or overwrite).
    Returns: {success, filepath, lines_written, error}
    """
    filepath = os.path.expanduser(filepath)

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        lines = content.count("\n") + 1
        logger.info(f"Wrote {lines} lines to {filepath}")

        return {"success": True, "filepath": filepath, "lines_written": lines, "error": None}
    except Exception as e:
        logger.error(f"Write failed for {filepath}: {e}")
        return {"success": False, "filepath": filepath, "lines_written": 0, "error": str(e)}


def apply_edit(filepath: str, original: str, replacement: str) -> dict:
    """
    Apply a search-and-replace edit to a file.
    Finds `original` text in the file and replaces it with `replacement`.

    Returns: {success, filepath, diff, error}
    """
    filepath = os.path.expanduser(filepath)

    read_result = read_file(filepath)
    if not read_result["success"]:
        return {"success": False, "filepath": filepath, "diff": None, "error": read_result["error"]}

    content = read_result["content"]

    # Find and replace
    if original not in content:
        # Try fuzzy match — strip whitespace differences
        original_stripped = original.strip()
        lines = content.split("\n")
        # Try to find the block by matching stripped lines
        orig_lines = original_stripped.split("\n")
        match_start = _fuzzy_find_block(lines, orig_lines)

        if match_start is not None:
            # Found it with fuzzy matching — replace the actual lines
            match_end = match_start + len(orig_lines)
            actual_original = "\n".join(lines[match_start:match_end])
            new_content = content.replace(actual_original, replacement)
        else:
            return {
                "success": False,
                "filepath": filepath,
                "diff": None,
                "error": "Could not find the text to replace in the file.",
            }
    else:
        new_content = content.replace(original, replacement, 1)

    # Generate diff for display
    diff = _generate_diff(content, new_content, filepath)

    # Write
    write_result = write_file(filepath, new_content)
    if not write_result["success"]:
        return {"success": False, "filepath": filepath, "diff": diff, "error": write_result["error"]}

    logger.info(f"Applied edit to {filepath}")
    return {"success": True, "filepath": filepath, "diff": diff, "error": None}


def apply_patch(filepath: str, edits: list[dict]) -> dict:
    """
    Apply multiple edits to a file in sequence.
    Each edit: {"find": "...", "replace": "..."}

    Returns: {success, filepath, edits_applied, diff, error}
    """
    filepath = os.path.expanduser(filepath)
    read_result = read_file(filepath)
    if not read_result["success"]:
        return {"success": False, "filepath": filepath, "edits_applied": 0, "diff": None, "error": read_result["error"]}

    content = read_result["content"]
    original_content = content
    applied = 0

    for edit in edits:
        find_text = edit.get("find", "")
        replace_text = edit.get("replace", "")

        if not find_text:
            continue

        if find_text in content:
            content = content.replace(find_text, replace_text, 1)
            applied += 1
        else:
            # Try fuzzy
            lines = content.split("\n")
            orig_lines = find_text.strip().split("\n")
            match_start = _fuzzy_find_block(lines, orig_lines)
            if match_start is not None:
                match_end = match_start + len(orig_lines)
                actual = "\n".join(lines[match_start:match_end])
                content = content.replace(actual, replace_text, 1)
                applied += 1

    if applied == 0:
        return {"success": False, "filepath": filepath, "edits_applied": 0, "diff": None, "error": "No edits matched."}

    diff = _generate_diff(original_content, content, filepath)
    write_result = write_file(filepath, content)

    return {
        "success": write_result["success"],
        "filepath": filepath,
        "edits_applied": applied,
        "diff": diff,
        "error": write_result.get("error"),
    }


def list_directory(dirpath: str, max_depth: int = 2) -> dict:
    """
    List directory structure (for project understanding).
    Returns: {success, path, tree, error}
    """
    dirpath = os.path.expanduser(dirpath)

    if not os.path.isdir(dirpath):
        return {"success": False, "path": dirpath, "tree": None, "error": "Not a directory"}

    try:
        tree_lines = []
        _build_tree(dirpath, tree_lines, prefix="", max_depth=max_depth, current_depth=0)
        tree = "\n".join(tree_lines[:500])  # Cap at 500 lines

        return {"success": True, "path": dirpath, "tree": tree, "error": None}
    except Exception as e:
        return {"success": False, "path": dirpath, "tree": None, "error": str(e)}


def detect_project(project_path: str) -> dict:
    """
    Detect what kind of project is at a path and what commands can run it.
    Returns: {type, runners: {dev, start, build, test, install}, files}
    """
    project_path = os.path.expanduser(project_path)

    if not os.path.isdir(project_path):
        return {"type": "unknown", "runners": {}, "files": [], "error": "Path not found"}

    detected_files = []
    runners = {}

    for config_file, commands in PROJECT_RUNNERS.items():
        full_path = os.path.join(project_path, config_file)
        if os.path.exists(full_path):
            detected_files.append(config_file)
            # Merge commands (later detections override)
            for key, cmd in commands.items():
                if key != "detect":
                    runners[key] = cmd

    # Try to detect more from package.json scripts
    pkg_json = os.path.join(project_path, "package.json")
    if os.path.exists(pkg_json):
        try:
            import json
            with open(pkg_json) as f:
                pkg = json.load(f)
            scripts = pkg.get("scripts", {})
            if "dev" in scripts:
                runners["dev"] = "npm run dev"
            if "start" in scripts:
                runners["start"] = "npm start"
            if "build" in scripts:
                runners["build"] = "npm run build"
            if "test" in scripts:
                runners["test"] = "npm test"
            if "lint" in scripts:
                runners["lint"] = "npm run lint"
        except Exception:
            pass

    project_type = "unknown"
    if "package.json" in detected_files:
        project_type = "node"
    elif "Cargo.toml" in detected_files:
        project_type = "rust"
    elif "pyproject.toml" in detected_files or "requirements.txt" in detected_files:
        project_type = "python"
    elif "go.mod" in detected_files:
        project_type = "go"
    elif "docker-compose.yml" in detected_files:
        project_type = "docker"
    elif "Makefile" in detected_files:
        project_type = "make"

    return {
        "type": project_type,
        "runners": runners,
        "files": detected_files,
        "path": project_path,
        "error": None,
    }


def find_file(name_pattern: str, search_path: str = "~/Documents") -> list[str]:
    """
    Find files matching a pattern. Used when user says 'edit main.py' without full path.
    Returns list of matching file paths (max 10).
    """
    search_path = os.path.expanduser(search_path)
    matches = []

    try:
        result = subprocess.run(
            ["find", search_path, "-name", name_pattern, "-type", "f",
             "-not", "-path", "*/node_modules/*",
             "-not", "-path", "*/.git/*",
             "-not", "-path", "*/__pycache__/*",
             "-not", "-path", "*/.venv/*"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout:
            matches = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
            matches = matches[:10]
    except Exception as e:
        logger.error(f"File search failed: {e}")

    return matches


# ─── Internal helpers ───

def _detect_language(filepath: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "tsx", ".jsx": "jsx", ".rs": "rust", ".go": "go",
        ".java": "java", ".rb": "ruby", ".php": "php", ".c": "c",
        ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".swift": "swift",
        ".kt": "kotlin", ".sh": "bash", ".zsh": "bash",
        ".yml": "yaml", ".yaml": "yaml", ".json": "json",
        ".toml": "toml", ".md": "markdown", ".html": "html",
        ".css": "css", ".scss": "scss", ".sql": "sql",
        ".dockerfile": "dockerfile", ".env": "env",
    }
    ext = Path(filepath).suffix.lower()
    name = Path(filepath).name.lower()

    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"

    return ext_map.get(ext, "text")


def _fuzzy_find_block(file_lines: list[str], search_lines: list[str]) -> Optional[int]:
    """
    Find a block of text in a file using fuzzy/stripped matching.
    Returns the starting line index or None.
    """
    search_stripped = [l.strip() for l in search_lines if l.strip()]
    if not search_stripped:
        return None

    for i in range(len(file_lines) - len(search_stripped) + 1):
        window = [file_lines[i + j].strip() for j in range(len(search_stripped))]
        # Check if the stripped versions match
        if window == search_stripped:
            return i

    # Partial match — try matching first and last lines
    first = search_stripped[0]
    last = search_stripped[-1]
    for i in range(len(file_lines)):
        if file_lines[i].strip() == first:
            end_idx = i + len(search_stripped) - 1
            if end_idx < len(file_lines) and file_lines[end_idx].strip() == last:
                return i

    return None


def _generate_diff(old_content: str, new_content: str, filepath: str) -> str:
    """Generate a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{os.path.basename(filepath)}",
        tofile=f"b/{os.path.basename(filepath)}",
        lineterm=""
    )
    return "".join(diff)[:5000]  # Cap diff output


def _build_tree(path: str, lines: list, prefix: str, max_depth: int, current_depth: int):
    """Build a directory tree representation."""
    if current_depth > max_depth:
        return

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    # Filter out noise
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".DS_Store", ".next", "dist", "build"}
    entries = [e for e in entries if e not in skip]

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{entry}")

        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            _build_tree(full_path, lines, prefix + extension, max_depth, current_depth + 1)
