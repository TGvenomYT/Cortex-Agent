"""
safety.py — the one gate every action Cortex takes goes through.

Chat, scheduled tasks, helpers and autonomy all ask `evaluate()`, so no path can be
looser than another. It answers one of three things:

    allow   run it
    ask     hold it for the user (live chat: approval buttons; background: approvals inbox)
    block   never run it, even if the user approves

Shell commands are judged by an ALLOW-LIST, not a list of bad words: a command runs on
its own only if every part of it is known to be read-only. Anything else is held, and
anything that can't be understood (command substitution, unparseable quoting) is held
too. This is what a denylist can't give you: `find -delete`, `python -c`, `curl | sh`
and everything nobody thought of fail closed.

"Tainted" means the turn has read outside content (a web page, an email, an uploaded
file, an MCP tool result). That text can contain instructions written by a stranger, so
from then on nothing that changes state runs without the user, and "allow in this chat"
grants stop applying.

Modes (Settings → Background → Approvals):
    careful   the allow-list above (default)
    balanced  ordinary commands also run on their own when the user drove the turn and it
              isn't tainted. Anything destructive is still held.
"""
from __future__ import annotations

import glob
import os
import re
import shlex
from dataclasses import dataclass

ALLOW, ASK, BLOCK = "allow", "ask", "block"
MODES = ("careful", "balanced")
# How worrying one simple command is, least to most.
_SAFE, _UNKNOWN, _DANGER, _BLOCKED = 0, 1, 2, 3
_MAX_DEPTH = 4


@dataclass(frozen=True)
class Verdict:
    kind: str               # allow | ask | block
    message: str = ""       # what the user sees: the draft, or why it was stopped
    grant: str = ""         # what "allow in this chat" would cover; "" = it can't be granted


# ── what's known to be read-only ──────────────────────────────────────────
_READ_ONLY = {
    # files + text (paths are checked separately, see _SENSITIVE)
    "ls", "ll", "pwd", "cat", "head", "tail", "wc", "file", "stat", "du", "df", "tree", "less", "more",
    "grep", "egrep", "fgrep", "rg", "sort", "uniq", "cut", "tr", "diff", "cmp", "md5", "md5sum",
    "shasum", "sha256sum", "basename", "dirname", "realpath", "readlink", "jq", "column", "nl", "rev", "tac",
    # system state
    "whoami", "id", "groups", "date", "cal", "uptime", "uname", "hostname", "sw_vers", "ps", "pgrep",
    "vm_stat", "iostat", "free", "lscpu", "lsblk", "system_profiler", "ioreg", "lsof", "netstat", "ss",
    "ifconfig", "ip", "arp", "route", "ping", "dig", "nslookup", "host", "traceroute", "which", "whereis",
    "type", "echo", "printf", "true", "false", "test", "[", "sleep", "seq", "bc", "wait", "cd",
    # Windows (cmd.exe)
    "dir", "ver", "where", "ipconfig", "systeminfo", "tasklist", "hostname.exe",
}
# program -> subcommands that only read
_READ_ONLY_SUB = {
    "git": {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "shortlog", "describe",
            "branch", "remote", "tag", "stash", "reflog", "config", "ls-remote", "cat-file", "rev-list",
            "merge-base", "whatchanged", "grep"},
    "docker": {"ps", "logs", "stats", "images", "inspect", "version", "info", "top", "port", "diff", "history"},
    "gh": {"repo", "pr", "issue", "run", "release", "status", "auth", "search"},
    "adb": {"devices", "version", "get-state", "get-serialno"},
    "brew": {"list", "ls", "info", "outdated", "config", "search", "deps", "leaves", "doctor"},
    "pip": {"list", "show", "freeze", "check", "--version"},
    "pip3": {"list", "show", "freeze", "check", "--version"},
    "npm": {"ls", "list", "view", "outdated", "--version", "-v", "root", "bin", "config"},
    "launchctl": {"list", "print", "print-disabled"},
    "ollama": {"list", "ps", "show", "--version", "-v"},
    "tailscale": {"status", "ip", "version", "ping"},
    "systemctl": {"status", "list-units", "list-timers", "is-active", "is-enabled", "show"},
    "journalctl": None,
    "diskutil": {"list", "info", "apfs"},
    "kubectl": {"get", "describe", "logs", "version", "top"},
    "flutter": {"--version", "doctor", "devices"},
    "tmutil": {"listbackups", "status", "latestbackup", "listlocalsnapshots"},
    "sysctl": None,
    "pmset": None,
    "defaults": {"read", "read-type", "domains"},
    "top": None,
}
# Subcommands of a read-only subcommand that still change things.
_GIT_WRITE_ARGS = {
    "branch": {"-d", "-D", "-m", "-M", "--delete", "--move", "-c", "-C"},
    "tag": {"-d", "--delete", "-a", "-s", "-f"},
    "remote": {"add", "remove", "rm", "rename", "set-url", "prune"},
    "stash": {"drop", "clear", "pop", "apply", "push", "save", "branch"},
    "config": {"--global", "--system", "--add", "--unset", "--replace-all"},
}
_VERSION_FLAGS = {"--version", "-V", "version", "--help", "-h"}
_GH_WRITE = {"create", "delete", "merge", "close", "reopen", "edit", "comment", "review", "fork",
             "clone", "rerun", "cancel", "login", "logout", "token", "sync", "archive"}

# Destructive or state-changing whatever the arguments.
_DANGEROUS = {
    "rm", "rmdir", "unlink", "shred", "srm", "mv", "dd", "truncate", "kill", "pkill", "killall", "chmod",
    "chown", "chflags", "chgrp", "launchctl", "crontab", "scutil", "networksetup", "hdiutil", "mount",
    "umount", "osascript", "scp", "rsync", "sftp", "nc", "ncat", "socat", "reg", "del", "erase", "rd",
    "rmdir.exe", "taskkill", "format", "diskpart", "sc", "bcdedit", "powershell", "pwsh", "systemctl",
    "service", "iptables", "ufw", "pfctl", "tccutil", "csrutil", "spctl", "defaults", "pmset", "tmutil",
    "diskutil", "kubectl", "git", "docker",
}
_SQL_CLIENTS = {"psql", "mysql", "mariadb", "sqlite3", "mongosh", "redis-cli"}
_SQL_WRITE = re.compile(r"\b(drop|delete|truncate|update|alter|insert|grant|revoke)\b", re.I)
_INTERPRETERS = {"sh", "bash", "zsh", "dash", "fish", "ksh", "cmd", "python", "python3", "python2",
                 "perl", "ruby", "node", "php", "lua"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
_FETCHERS = {"curl", "wget", "fetch", "aria2c"}
_WRAPPERS = {"nohup", "nice", "ionice", "time", "command", "exec", "caffeinate", "stdbuf", "builtin"}
_SYSTEM_DIRS = {"/usr", "/etc", "/bin", "/sbin", "/var", "/opt", "/private", "/System", "/Library",
                "/Applications", "/Users", "/home", "/root", "/boot", "/lib"}

# Files that hold credentials. Reading one is not "just reading".
_SENSITIVE = re.compile(
    r"(^|[/\\])\.ssh([/\\]|$)|id_(rsa|ed25519|ecdsa|dsa)\b|(^|[/\\])\.aws([/\\]|$)|\.gnupg|\.netrc|\.npmrc"
    r"|\.pypirc|\.git-credentials|(^|[/\\])\.env(\.|$)|cortex\.env|\.cortex[/\\]secrets|secrets\.json"
    r"|Keychains?[/\\]|\.keychain|\.config[/\\]gh|\.docker[/\\]config\.json|\.kube[/\\]config|\.pgpass",
    re.I)


# ── pulling a command line apart ──────────────────────────────────────────
def _opaque(command: str) -> str:
    """Why this line can't be analysed statically, or ''. Tracks quotes: `$(x)` inside
    single quotes is literal text, inside double quotes it still runs."""
    quote = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote == "'":
            quote = None if c == "'" else quote
        elif c == "\\":
            i += 1
        elif c == '"':
            quote = None if quote == '"' else '"'
        elif c == "'":
            quote = "'"
        elif c == "`" or (c == "$" and command[i + 1:i + 2] == "("):
            return "it runs a nested command I can't check first"
        elif quote is None and c in "<>" and command[i + 1:i + 2] == "(":
            return "it runs a nested command I can't check first"
        i += 1
    return ""


def _tokens(command: str) -> list[str] | None:
    try:
        lex = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=";&|<>()")
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:      # unbalanced quotes
        return None


def _split(tokens: list[str]) -> tuple[list[list[list[str]]], list[str]]:
    """-> (pipelines, written_files). A pipeline is a list of simple commands; a simple
    command is its argv with redirections removed."""
    pipelines, pipeline, cur, writes = [], [], [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t and set(t) <= set("<>&|;()"):
            if ">" in t or "<" in t:
                target = tokens[i + 1] if i + 1 < len(tokens) else ""
                if cur and cur[-1].isdigit():
                    cur.pop()                   # the fd in `2>`
                i += 1
                if ">" in t and not re.fullmatch(r"-|\d+|/dev/null|/dev/stdout|/dev/stderr", target):
                    writes.append(target)
            elif t in ("|", "|&"):
                pipeline.append(cur)
                cur = []
            else:                               # ; && || & ( )
                pipeline.append(cur)
                pipelines.append(pipeline)
                pipeline, cur = [], []
        else:
            cur.append(t)
        i += 1
    pipeline.append(cur)
    pipelines.append(pipeline)
    cleaned = [[c for c in p if c] for p in pipelines]
    return [p for p in cleaned if p], writes


def _expand(path: str) -> str:
    home = os.path.expanduser("~")
    p = path.strip()
    for tilde in ("~", "$HOME", "${HOME}", "%USERPROFILE%"):
        if p == tilde or p.startswith(tilde + "/"):
            p = home + p[len(tilde):]
            break
    return p


def _wide_target(path: str) -> bool:
    """A delete target so broad it's refused outright: the disk, a home folder, a top-level
    system folder, or a folder directly under home (~/Documents)."""
    p = _expand(path)
    if p in ("*", ".", "..", "./*", "../*", "./", "../"):
        return True
    p = re.sub(r"/\*+$", "", p).rstrip("/") or "/"
    home = os.path.expanduser("~").rstrip("/")
    return p == "/" or p == home or os.path.dirname(p) == home or p in _SYSTEM_DIRS \
        or os.path.dirname(p) in ("/Users", "/home")


def _risky_write(path: str) -> bool:
    """Writing here changes how the machine or a program behaves, not just a file's contents."""
    p = _expand(path)
    home = os.path.expanduser("~").rstrip("/")
    return bool(_SENSITIVE.search(p) or re.search(r"Library/(LaunchAgents|LaunchDaemons)|\.config/", p)
                or (os.path.dirname(p) == home and os.path.basename(p).startswith("."))
                or any(p == d or p.startswith(d + "/") for d in _SYSTEM_DIRS - {"/private", "/Users", "/home", "/var"}))


def _rm_verdict(args: list[str], sudo: bool) -> tuple[int, str]:
    flags = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    recursive = "r" in flags.lower() or "--recursive" in args or "--dir" in args
    targets = [a for a in args if not a.startswith("-")]
    if sudo:
        return _BLOCKED, "it deletes files with administrator rights"
    if recursive and (not targets or any(_wide_target(t) for t in targets)):
        return _BLOCKED, "it would delete a whole folder tree there's no getting back"
    return _DANGER, "it deletes files"


def _after_flags(args: list[str]) -> list[str]:
    """What a wrapper (nohup, sudo, xargs…) runs: its first non-flag word and everything
    after it, flags included — `nohup rm -rf /` must still be judged as `rm -rf /`."""
    for i, a in enumerate(args):
        if not a.startswith("-"):
            return args[i:]
    return []


def _ssh_verdict(args: list[str], known_hosts: set, depth: int) -> tuple[int, str, str]:
    """Judge `ssh [opts] host command` by what runs on the far side, for a known host only."""
    i, host = 0, None
    with_value = set("bcDEeFIiJLlmOopQRSWw")
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            if a[-1] in with_value and len(a) == 2:
                i += 1
            i += 1
            continue
        host = a
        break
    remote = " ".join(args[i + 1:]) if host else ""
    name = (host or "").split("@")[-1]
    if not host:
        return _UNKNOWN, "it opens a remote connection", "ssh"
    if name not in known_hosts:
        return _DANGER, f"it connects to {name}, which isn't your configured server", ""
    if not remote.strip():
        return _DANGER, "it opens an interactive session on your server", ""
    level, why, key = _classify(remote, known_hosts, depth + 1)
    return level, (f"on your server, {why}" if why else ""), key and f"ssh:{key}"


def _simple(prog: str, args: list[str], raw: str, known_hosts: set, depth: int,
            sudo: bool = False) -> tuple[int, str, str]:
    """Judge one simple command. Returns (level, why, grant_key)."""
    name = prog.lower()
    if name.endswith(".exe"):
        name = name[:-4]

    # refused outright, whoever asks
    if name == "rm" or name == "del" or name == "rd":
        lvl, why = _rm_verdict(args, sudo)
        return lvl, why, ""
    if name.startswith("mkfs") or name == "fdisk":
        return _BLOCKED, "it would erase a disk", ""
    if name in ("shutdown", "reboot", "halt", "poweroff"):
        return _BLOCKED, "it would turn the machine off", ""
    if name == "dd" and any(a.startswith("of=/dev/") for a in args):
        return _BLOCKED, "it would write straight over a disk", ""
    if name == "diskutil" and args and re.match(r"(erase|reformat|partition|secureerase|apfs$)", args[0], re.I) \
            and not (args[0].lower() == "apfs" and args[1:2] and args[1].lower().startswith("list")):
        return _BLOCKED, "it would erase a disk", ""
    if name == "git" and "push" in args and any(
            a in ("--force", "-f", "--force-with-lease") or a.startswith("+") or a.startswith("--force")
            for a in args[args.index("push") + 1:]):
        return _BLOCKED, "a force-push rewrites history for everyone", ""
    if name == "security" and args[:1] and re.match(
            r"(find-|dump-|export|unlock|delete-|add-|import|set-)", args[0]):
        return _BLOCKED, "it reaches into your saved passwords", ""
    if name in ("chmod", "chown") and "-R" in args and any(a == "/" or _wide_target(a) for a in args[1:]):
        return _BLOCKED, "it would change permissions across the whole system", ""

    # wrappers: judge what they wrap
    if name in ("sudo", "doas"):
        rest = _after_flags(args)
        if not rest:
            return _DANGER, "it asks for administrator rights", ""
        lvl, why, _ = _simple(os.path.basename(rest[0]), rest[1:], raw, known_hosts, depth + 1, sudo=True)
        return max(lvl, _DANGER), why or "it runs with administrator rights", ""
    if name in _WRAPPERS:
        rest = _after_flags(args)
        if not rest:
            return _SAFE, "", ""
        return _simple(os.path.basename(rest[0]), rest[1:], raw, known_hosts, depth + 1, sudo)
    if name == "xargs":
        rest = _after_flags(args)
        if not rest:
            return _SAFE, "", ""
        lvl, why, key = _simple(os.path.basename(rest[0]), rest[1:], raw, known_hosts, depth + 1, sudo)
        return max(lvl, _SAFE if lvl == _SAFE else _DANGER), why or "it runs on a list of files", key
    if name in _SHELLS and "-c" in args:
        idx = args.index("-c")
        inner = " ".join(args[idx + 1:])
        if depth >= _MAX_DEPTH:
            return _DANGER, "it's nested too deeply to check", ""
        return _classify(inner, known_hosts, depth + 1)
    if name == "ssh":
        return _ssh_verdict(args, known_hosts, depth)

    # anything touching a credentials file is held, even with a "safe" program
    for a in args:
        if _SENSITIVE.search(a):
            return _DANGER, "it reads a file that holds private credentials", ""

    if name == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        if sub == "push" or sub == "reset" or sub == "clean" or sub == "filter-branch" or sub == "rebase":
            return _DANGER, f"`git {sub}` can change or lose work", ""
        if set(args) & _GIT_WRITE_ARGS.get(sub, set()):
            return _DANGER, f"that `git {sub}` changes your repo", ""
        if sub in _READ_ONLY_SUB["git"]:
            if sub == "config" and not any(a in ("--get", "--list", "-l") for a in args):
                return _UNKNOWN, "", "git config"
            return _SAFE, "", ""
        return _UNKNOWN, "", f"git {sub}".strip()
    if name == "gh":
        return (_UNKNOWN, "", "gh") if (set(args) & _GH_WRITE or "api" in args[:1]) else (_SAFE, "", "")
    if name == "docker":
        sub = args[0] if args else ""
        if sub == "compose":
            sub2 = args[1] if len(args) > 1 else ""
            if sub2 in ("ps", "logs", "config", "ls", "top", "version"):
                return _SAFE, "", ""
            return _DANGER if sub2 in ("down", "rm", "kill", "stop") else _UNKNOWN, \
                f"`docker compose {sub2}` stops or removes containers" if sub2 in ("down", "rm", "kill", "stop") else "", \
                "docker compose"
        if sub in ("rm", "rmi", "stop", "kill", "prune", "system", "volume", "network", "container", "image"):
            return _DANGER, f"`docker {sub}` can stop or remove things", ""
        return (_SAFE, "", "") if sub in _READ_ONLY_SUB["docker"] else (_UNKNOWN, "", "docker")
    if name == "find":
        if set(args) & {"-delete", "-fprint", "-fprintf", "-fls"}:
            return _DANGER, "it can delete or change the files it finds", ""
        for flag in ("-exec", "-execdir", "-ok", "-okdir"):
            if flag in args:        # judged by what it runs on each file: `-exec wc -l {} +` only reads
                inner = args[args.index(flag) + 1:]
                inner = [a for a in inner if a not in ("{}", "+", ";")]
                if not inner:
                    return _DANGER, "it can delete or change the files it finds", ""
                level, why, key = _simple(os.path.basename(inner[0]), inner[1:], raw, known_hosts, depth + 1, sudo)
                if level != _SAFE:
                    return max(level, _DANGER), why or "it can delete or change the files it finds", ""
        return _SAFE, "", ""
    if name == "sed":
        if any(a.startswith("-i") or a == "--in-place" for a in args):
            return _DANGER, "it edits files in place", ""
        script = " ".join(a for a in args if not a.startswith("-"))
        return (_UNKNOWN, "", "sed") if re.search(r"(^|[;{}/])\s*[we]\b", script) else (_SAFE, "", "")
    if name in ("awk", "gawk", "mawk"):
        return (_UNKNOWN, "", "awk") if re.search(r"system|getline|[|>]", " ".join(args)) else (_SAFE, "", "")
    if name == "xcrun" or name == "open":
        return _UNKNOWN, "", name
    if name in _SQL_CLIENTS:
        return (_DANGER, "it may change a database", "") if _SQL_WRITE.search(raw) else (_UNKNOWN, "", name)
    if name in _READ_ONLY:
        return _SAFE, "", ""
    if name in _READ_ONLY_SUB:
        allowed = _READ_ONLY_SUB[name]
        sub = next((a for a in args if not a.startswith("-")), args[0] if args else "")
        if allowed is None:
            if name == "sysctl" and any("=" in a or a == "-w" for a in args):
                return _DANGER, "it changes a system setting", ""
            if name == "pmset" and "-g" not in args:
                return _DANGER, "it changes power settings", ""
            return _SAFE, "", ""
        if sub in allowed or not args:
            return _SAFE, "", ""
    if name in _INTERPRETERS or name in ("pip", "pip3", "npm", "npx", "cargo", "go", "java"):
        if args and set(args) <= _VERSION_FLAGS:
            return _SAFE, "", ""
    if name in _DANGEROUS:
        return _DANGER, f"`{name}` can change or remove things", ""
    if args and set(args) <= _VERSION_FLAGS:
        return _SAFE, "", ""
    return _UNKNOWN, "", name


def _classify(command: str, known_hosts: set, depth: int = 0) -> tuple[int, str, str]:
    """(level, why, grant_key) for a whole command line — the worst of its parts."""
    if depth > _MAX_DEPTH:
        return _DANGER, "it's nested too deeply to check", ""
    why = _opaque(command)
    if why:
        return _DANGER, why, ""
    tokens = _tokens(command)
    if tokens is None:
        return _DANGER, "I can't read its quoting", ""
    pipelines, writes = _split(tokens)
    worst, worst_why, key = _SAFE, "", ""

    def note(level, reason, k=""):
        nonlocal worst, worst_why, key
        if level > worst or (level == worst and not key and k):
            worst, worst_why, key = level, reason, k

    for path in writes:
        if re.match(r"/dev/(sd|disk|rdisk|nvme)", path):
            note(_BLOCKED, "it would write straight over a disk")
        elif _risky_write(path):
            note(_DANGER, f"it writes to {path}")
        else:
            note(_UNKNOWN, "", "writing files")
    for pipeline in pipelines:
        names = []
        for argv in pipeline:
            i = 0
            while i < len(argv) and re.match(r"^[A-Za-z_]\w*=", argv[i]):
                i += 1
            if i >= len(argv):
                continue
            if i:
                note(_UNKNOWN, "", "env")   # FOO=1 prog: the prefix can change what prog does
            prog, args = os.path.basename(argv[i]), argv[i + 1:]
            names.append(prog.lower())
            note(*_simple(prog, args, " ".join(argv), known_hosts, depth))
        fetched = any(n in _FETCHERS for n in names)
        if fetched and any(n in _INTERPRETERS for n in names):
            note(_DANGER, "it downloads a script and runs it straight away", "")
    return worst, worst_why, key


def _known_hosts() -> set:
    hosts = {"localhost", "127.0.0.1", "myserver"}
    try:
        import cortex.storage.settings_store as settings_store
        host = (settings_store.load().get("server") or {}).get("host")
        if host:
            hosts.add(host)
    except Exception:
        pass
    return hosts


def classify_shell(command: str) -> tuple[str, str, str]:
    """-> ('safe'|'unknown'|'danger'|'block', why, grant_key) for a shell command."""
    level, why, key = _classify(command or "", _known_hosts())
    return ("safe", "unknown", "danger", "block")[level], why, key


def write_targets(command: str, cwd: str = None) -> tuple[list, bool]:
    """What a shell command will change on this machine, for keeping an undo copy.
    -> (paths, complete). complete is False if it may also change things that can't be
    told from its text: a script, an installer, anything on another machine."""
    tokens = _tokens(command or "")
    if tokens is None or _opaque(command or ""):
        return [], False
    pipelines, writes = _split(tokens)
    here = cwd or os.getcwd()
    paths, complete = [], True

    def at(p):
        return os.path.normpath(os.path.join(here, _expand(p)))

    def add(p):
        hits = glob.glob(at(p)) if any(c in p for c in "*?[") else []
        paths.extend(hits or [at(p)])

    for pipeline in pipelines:
        for argv in pipeline:
            i = 0
            while i < len(argv) and re.match(r"^[A-Za-z_]\w*=", argv[i]):
                i += 1
            argv = argv[i:]
            while argv and os.path.basename(argv[0]).lower() in _WRAPPERS:
                argv = _after_flags(argv[1:])
            if not argv:
                continue
            prog, args = os.path.basename(argv[0]).lower(), argv[1:]
            plain = [a for a in args if not a.startswith("-")]
            if prog == "cd":
                here = at(plain[0]) if plain else os.path.expanduser("~")
            elif _simple(prog, args, " ".join(argv), set(), 0)[0] == _SAFE:
                continue
            elif prog in ("rm", "unlink", "rmdir", "truncate", "shred", "touch", "mkdir", "tee"):
                for p in plain:
                    if not (prog == "truncate" and p.isdigit()):
                        add(p)
            elif prog == "mv" and len(plain) >= 2:
                dest = at(plain[-1])
                for src in plain[:-1]:
                    add(src)
                    paths.append(os.path.join(dest, os.path.basename(src)) if os.path.isdir(dest) else dest)
            elif prog == "cp" and len(plain) >= 2:
                dest = at(plain[-1])
                for src in plain[:-1]:
                    paths.append(os.path.join(dest, os.path.basename(src)) if os.path.isdir(dest) else dest)
            elif prog == "ln" and plain:
                add(plain[-1])
            elif prog == "sed" and len(plain) >= 2:
                for p in plain[1:]:
                    add(p)
            elif prog in ("chmod", "chown", "chgrp") and len(plain) >= 2:
                for p in plain[1:]:
                    add(p)
            else:
                complete = False
    for path in writes:
        add(path)
    return paths, complete


def is_catastrophic(command: str) -> tuple[bool, str]:
    """Commands refused outright, even with the user's approval."""
    level, why, _ = classify_shell(command)
    if level == "block":
        return True, f"`{command}` — {why}, so I won't run that."
    return False, ""


# ── the gate ──────────────────────────────────────────────────────────────
def _shell_verdict(cmd: str, tainted: bool, background: bool, grants, mode: str) -> Verdict:
    level, why, key = classify_shell(cmd)
    show = f"I'd like to run this:\n\n```\n{cmd}\n```\n"
    if level == "block":
        return Verdict(BLOCK, f"I won't run `{cmd}` — {why}.")
    if level == "safe":
        return Verdict(ALLOW)
    if tainted:
        return Verdict(ASK, show + "\n**Check this one.** It came up right after I read outside content, so that "
                                   "content could be what's asking for it. Only say yes if you wanted this.")
    if level == "danger":
        return Verdict(ASK, show + f"\nHeads up: {why}. Go ahead?")
    if not background and (mode == "balanced" or (key and key in grants)):
        return Verdict(ALLOW)
    return Verdict(ASK, show + "\nOK to run it?", key)


def evaluate(action: dict, *, tainted: bool = False, background: bool = False,
             grants=frozenset(), mode: str = "careful", mcp_readonly: bool = False) -> Verdict:
    """Decide what happens to one proposed action. See the module docstring."""
    kind = action.get("action", "")
    effective_grants = frozenset() if (tainted or background) else frozenset(grants or ())

    if kind in ("execute", "devops"):
        cmd = action.get("command") or ""
        return _shell_verdict(cmd, tainted, background, effective_grants, mode) if cmd else Verdict(ALLOW)

    if kind in ("answer", "ask", "search", "image_search", "email_check", "email_search", "delegate",
                "credential_prompt"):
        return Verdict(ALLOW)

    if kind == "browse":
        if action.get("browse_open") and tainted:
            return Verdict(ASK, f"Open {action.get('browse_url')} in your browser?")
        return Verdict(ALLOW)

    if kind == "browser":
        steps = action.get("browser_steps") or []
        acting = [s for s in steps if (s or {}).get("action") in ("fill", "click", "screenshot")]
        if acting and tainted:
            return Verdict(ASK, f"I'd like to interact with {action.get('browser_url')} in a real browser "
                                "(typing or clicking). OK?")
        return Verdict(ALLOW)

    if kind == "crawl":
        cred = action.get("crawl_credential")
        if cred:
            return Verdict(ASK, f"I'd like to log in to {action.get('crawl_url')} with your saved "
                                f"`{cred}` login. OK?")
        return Verdict(ALLOW)

    if kind == "chart":
        import cortex.llm.viz as viz
        ok, why = viz.check_code(action.get("code") or "")
        if ok:
            return Verdict(ALLOW)
        return Verdict(ASK, f"This chart's code does more than draw a chart ({why}):\n\n"
                            f"```python\n{(action.get('code') or '')[:1500]}\n```\n\nRun it anyway?")

    if kind == "edit":
        msg = (f"I'd like to edit `{action.get('edit_file')}`: {action.get('edit_instruction', '')}. Go ahead?")
        return Verdict(ALLOW) if "edit" in effective_grants else Verdict(ASK, msg, "edit")

    if kind == "run_project":
        path, op = action.get("run_project_path", "."), action.get("run_project_command", "")
        key = f"run_project:{path}"
        if key in effective_grants:
            return Verdict(ALLOW)
        return Verdict(ASK, f"I'd like to run the project's `{op}` step in `{path}`. OK?", key)

    if kind == "email_send":
        to = action.get("email_to") or "(no recipient)"
        subject = action.get("email_subject") or "(no subject)"
        body = (action.get("email_body") or "").strip()
        draft = f"To: {to}\nSubject: {subject}" + (f"\n\n{body}" if body else "")
        # Unconditional — there is no "obvious" email that's safe to fire off without
        # the user seeing it first, no matter how clearly they asked.
        return Verdict(ASK, f"Here's the draft — nothing has been sent yet:\n\n{draft}\n\nSend it?")

    if kind == "remind":
        r = action.get("reminder") or {}
        if r.get("op") == "delete":
            return Verdict(ASK, f"Delete reminder #{r.get('id')}?")
        if r.get("op") != "list" and tainted:
            return Verdict(ASK, f"Set a reminder: {r.get('text')}?")
        return Verdict(ALLOW)

    if kind == "schedule_task":
        op = action.get("task_op", "add")
        if op == "list":
            return Verdict(ALLOW)
        if op == "remove":
            return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Remove scheduled task {action.get('task_id')}?")
        return Verdict(ASK, f"I'd like to schedule this to run on its own, {_when(action.get('task_schedule'))}:\n\n"
                            f"> {action.get('task_prompt')}\n\nIt keeps running in the background until "
                            "it finishes or you stop it. Set it up?")

    if kind == "workers":
        op, w, prof = action.get("workers_op"), action.get("worker"), action.get("profile")
        if op in ("list", "unload_model"):
            return Verdict(ALLOW)
        if op == "disable" or (op == "profile" and prof in ("off", "essential")):
            # Switching work off is cheap, but outside text must not get to do it silently.
            return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Turn off `{w or prof}`?")
        if op == "enable":
            return Verdict(ASK, f"Turn on the background worker `{w}`? It will keep running on its "
                                "schedule until you switch it off.")
        if op == "run":
            return Verdict(ASK, f"Run the background worker `{w}` now?")
        if op == "profile":
            return Verdict(ASK, f"Switch background work to the '{prof}' profile? That starts several "
                                "scheduled jobs" + (", including the AI model." if prof == "full" else "."))
        return Verdict(ASK, f"Change background workers ({op})?")

    if kind == "remember":
        content = action.get("remember_content", "")
        if action.get("remember_type") == "rule":
            return Verdict(ASK, f"Save this as a standing rule I follow from now on?\n\n> {content}")
        return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Remember this?\n\n> {content}")

    if kind == "forget":
        target = action.get("forget_search") or f"#{action.get('forget_id')}"
        return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Forget {target}?")

    if kind == "watch":
        op = action.get("watch_op", "add")
        if op == "list":
            return Verdict(ALLOW)
        if op == "stop":
            return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Stop watch {action.get('watch_id')}?")
        cmd = action.get("watch_command")
        if cmd and classify_shell(cmd)[0] != "safe":
            return Verdict(ASK, f"I'd like to keep checking this:\n\n```\n{cmd}\n```\n\nIt isn't read-only, so OK?")
        return Verdict(ALLOW) if not tainted else Verdict(ASK, f"Keep an eye on {action.get('watch_label')}?")

    if kind == "undo":
        return Verdict(ASK, "Undo the changes from my last job in this chat?")

    if kind == "mcp":
        name = action.get("mcp_name", "")
        if mcp_readonly and not tainted:
            return Verdict(ALLOW)
        key = f"mcp:{name}"
        if key in effective_grants:
            return Verdict(ALLOW)
        return Verdict(ASK, f"I'd like to use `{name.replace('mcp__', '').replace('__', ' · ')}` with:\n\n"
                            f"```\n{str(action.get('mcp_args') or {})[:600]}\n```\n\nOK?", key)

    return Verdict(ASK, f"I'd like to do something I can't classify ({kind or 'unknown'}). OK?")


def _when(schedule) -> str:
    """What the schedule will really do, which may be slower than what was asked for."""
    from cortex.core.tasks import parse_schedule
    try:
        return parse_schedule(schedule)[1]
    except ValueError:
        return str(schedule)


def needs_confirmation(action: dict) -> tuple[bool, str]:
    """True if `action` can't just run. Kept for callers that don't care about the details."""
    v = evaluate(action)
    return v.kind != ALLOW, v.message


def current_mode() -> str:
    try:
        import cortex.storage.settings_store as settings_store
        m = settings_store.section("agents").get("approvals", "careful")
        return m if m in MODES else "careful"
    except Exception:
        return "careful"
