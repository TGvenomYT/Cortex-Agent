"""
Tool definitions for native LLM tool-calling — the top-level "what should I
do" decision, as opposed to schemas.py's structured-extraction schemas.

Native tool-calling (Ollama's `tools` param) uses the model's actual
tool-use training instead of asking it to author a bespoke JSON envelope in
prose. Confirmed empirically: this is what actually fixes the model
inventing wrong action types / shapes — it's not fighting its own priors
against a custom format anymore.

tool_call_to_action_data() maps a returned tool call into the exact same
dict shape the rest of cortex.py already expects from the old JSON-envelope
response (action/command/edit_file/reminder/etc.) — that's what keeps every
downstream consumer (safety.needs_confirmation, the per-action execution
branches, experience logging, _pending_actions) unchanged.
"""
import json
import re
import uuid

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": (
                "Run a shell command on the local macOS machine, or via SSH on the "
                "remote server. Use this for ANY question about current local/repo/"
                "container/system state — you have no passive knowledge of it, you "
                "must run a command to find out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the exact shell command to run"},
                    "background": {"type": "boolean", "description": (
                        "true for anything long-running (a server, a big build or install). It's started "
                        "and watched for you, and you report back here automatically when it ends, so "
                        "don't wait or poll. Anything still going after a minute moves there by itself.")},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information you don't already know, or any time-sensitive/current-events info.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetch and read a web page's content, or open it in the user's default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "open_in_browser": {"type": "boolean", "description": "true to open in the browser, false to fetch and read the content yourself"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_automate",
            "description": (
                "Drive a REAL browser (headless Chromium) to load a JavaScript-heavy page "
                "that static fetch can't read, or to interact: log in, fill forms, click, "
                "scroll, wait for content, extract rendered data, screenshot. Use this when "
                "browse_url returns little/empty content or the task needs clicking/typing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "description": ("optional interaction steps in order; each is one of: "
                                        "{action:'fill',selector,value} {action:'click',selector} "
                                        "{action:'wait',selector|ms} {action:'extract',selector} "
                                        "{action:'screenshot',path}. Omit to just read the rendered page."),
                        "items": {"type": "object"},
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_site",
            "description": "Navigate a website across multiple pages, optionally using a saved login credential, to gather information (e.g. authenticated dashboards).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "credential": {"type": ["string", "null"], "description": "name of a configured credential, or null for public sites"},
                    "max_pages": {"type": "integer"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Modify a file's contents. The user approves it first; once they do, your job carries on from here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "instruction": {"type": "string", "description": "what to change"},
                },
                "required": ["path", "instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_project",
            "description": "Run a project's standard step. Works out how from the folder (npm, Python, Cargo, Go, Flutter, Make). dev/start run in the background and return a log path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "command": {"type": "string", "enum": ["dev", "start", "build", "test", "install"]},
                },
                "required": ["path", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Generate a graph/plot/chart to visualize math or data, shown inline. "
                "Provide matplotlib Python code that builds a figure — `plt`, `np` (numpy) and "
                "`pd` (pandas, may be None) are available; do NOT import them or call savefig. "
                "Use for ANY 'plot/graph/chart/visualize/draw' request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "matplotlib code using plt/np"},
                    "caption": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Fetch real images from the web to show the user (e.g. 'show me pictures of X', 'find images of Y').",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "When you need the user to make a decision or clarify something, ASK instead of "
                "guessing. Give 2-4 concrete options to choose from (they can also type their own)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"},
                                "description": "2-4 short choices"},
                },
                "required": ["question", "options"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_check",
            "description": "Check the user's unread Gmail across all their connected accounts. Use for 'any new/important emails', 'check my inbox', morning briefings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_search",
            "description": "Search the user's connected Gmail accounts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_send",
            "description": "Send an email from the user's Gmail. ONLY when the user explicitly asks to send/reply. Always summarize what you're sending in your reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "from": {"type": ["string", "null"], "description": "which connected account to send from; null = first"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_reminder",
            "description": "Create, list, delete, or edit a reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["create", "list", "delete", "edit"]},
                    "text": {"type": ["string", "null"]},
                    "due": {"type": ["string", "null"], "description": "ISO 8601 timestamp, when op=create or edit"},
                    "repeat": {"type": ["string", "null"], "description": "null, or one of: daily, weekly, weekdays"},
                    "id": {"type": ["integer", "null"], "description": "reminder id, when op=delete or edit"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": (
                "Set up a RECURRING background task that runs on a schedule and delivers "
                "its result to the user (e.g. 'every morning at 8 email me a news briefing', "
                "'every hour check if my server is healthy and notify me'). Different from "
                "manage_reminder (a one-off nudge): this re-runs a full prompt through you on "
                "a cadence, so each run wakes the AI model — keep it as infrequent as the job allows "
                "(the minimum is every 5 min). A job with an end, like 'tell me when X is done', stops "
                "itself once it's finished. Use op='add' with desc, prompt, schedule; op='list'; "
                "op='remove' with id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "list", "remove"]},
                    "desc": {"type": ["string", "null"], "description": "short label for the task"},
                    "prompt": {"type": ["string", "null"], "description": "the instruction to run each time"},
                    "schedule": {"type": ["string", "null"],
                                 "description": "e.g. 'daily 08:00', 'at 5pm', 'every 30 min', 'hourly', 'every friday at 9am', 'every 2 days'"},
                    "deliver": {"type": ["string", "null"], "enum": ["notify", "email", "both", None],
                                "description": "how to deliver the result; default notify"},
                    "id": {"type": ["string", "null"], "description": "task id, when op=remove"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "watch_process",
            "description": (
                "Look after something that's ALREADY running and come back to this chat by yourself the "
                "moment something happens, without the user asking. Use for 'tell me when my build "
                "finishes', 'keep an eye on this server', 'let me know if that crashes'. It costs nothing "
                "while it waits. Give exactly one target: pid, process_name, port, file or check_command. "
                "Prefer this to schedule_task for anything already running, and never poll in a loop. "
                "op: add | list | stop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "list", "stop"]},
                    "label": {"type": ["string", "null"], "description": "short name for what's being watched"},
                    "note": {"type": ["string", "null"], "description": "what the user wants to know"},
                    "pid": {"type": ["integer", "null"]},
                    "process_name": {"type": ["string", "null"], "description": "matched against running commands"},
                    "port": {"type": ["integer", "null"]},
                    "file": {"type": ["string", "null"], "description": "a file to wait for, or a log to read"},
                    "pattern": {"type": ["string", "null"], "description": "with file: the text to look for"},
                    "check_command": {"type": ["string", "null"], "description": "a read-only command to repeat"},
                    "until": {"type": ["string", "null"], "description": (
                        "port: open | closed. check_command: success | fail | changes | contains:<text> | gone:<text>")},
                    "id": {"type": ["string", "null"], "description": "watch id, for op=stop"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_changes",
            "description": "Undo what the last job in this chat changed on disk, when the user asks to undo or revert it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Permanently store a fact, preference, or standing rule the moment it comes "
                "up — do NOT wait, and do not assume you'll pick it up passively later. Use "
                "type='rule' for a standing instruction about how you must behave going "
                "forward (e.g. the user says 'remember:', 'always...', 'never...', or "
                "corrects something you did) — rules are never pruned and are always included "
                "in your instructions, every single turn. Use type='fact'/'preference'/"
                "'project_info' for ordinary things worth remembering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": ["fact", "preference", "rule", "project_info"]},
                },
                "required": ["content", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete a previously stored memory or rule, e.g. when the user says to forget it, it's no longer true, or they want a standing rule lifted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": ["integer", "null"], "description": "the memory's id, if known"},
                    "search": {"type": ["string", "null"], "description": "a snippet of its content, if the id isn't known"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Hand parts of a big job to helper agents that each work on one part with a focused "
                "context. Use ONLY when the request has 2+ clearly INDEPENDENT parts (e.g. 'check my "
                "disk, search the web for X, and summarise my unread mail'). NEVER for a single "
                "question, a simple task, chit-chat, or steps that depend on each other's results. "
                "Make every subtask fully self-contained."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "description": "2-4 independent parts",
                        "items": {"type": "object",
                                  "properties": {"title": {"type": "string", "description": "3-5 words"},
                                                 "prompt": {"type": "string", "description": "complete instruction for the helper"}},
                                  "required": ["title", "prompt"]},
                    },
                },
                "required": ["subtasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_workers",
            "description": (
                "Control Cortex's OWN background workers — the scheduled jobs that run when the user "
                "isn't chatting (reminders, system collectors, AI autonomy, daily briefings, sync, and "
                "recurring tasks) — and the AI model's memory use. Use for 'what's running in the "
                "background', 'stop/pause the background workers', 'turn off autonomy', 'free up "
                "memory'. op: list | enable | disable | profile | run | unload_model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["list", "enable", "disable", "profile", "run", "unload_model"]},
                    "worker": {"type": ["string", "null"],
                               "description": "worker key or name from op=list (autonomy, reminders, collector.system, "
                                              "report-morning, task.<id>…), for enable/disable/run"},
                    "profile": {"type": ["string", "null"], "enum": ["off", "essential", "balanced", "full", None],
                                "description": "preset for op=profile: off = nothing runs, essential = reminders+rules only, "
                                               "balanced = + machine awareness, full = + AI autonomy/briefings"},
                },
                "required": ["op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_credentials",
            "description": "Trigger a native macOS dialog asking the user for login credentials to a site you need to access.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {"type": "string"},
                    "purpose": {"type": "string", "description": "what you'll do once you have the credentials"},
                },
                "required": ["site", "purpose"],
            },
        },
    },
]


_INLINE_CALL = re.compile(r"\{.*\}", re.S)


def rescue_tool_calls(result: dict, known: set) -> dict:
    """Some models write a tool call as JSON in their reply instead of using the tool
    channel, which would otherwise show the user raw JSON and do nothing. If the whole
    reply is exactly one call to a tool we offer, treat it as that call."""
    if result.get("tool_calls"):
        return result
    text = (result.get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    text = re.sub(r"^<tool_call>\s*|\s*</tool_call>$", "", text).strip()
    if not _INLINE_CALL.fullmatch(text):
        return result
    try:
        call = json.loads(text)
    except ValueError:
        return result
    if isinstance(call, dict) and isinstance(call.get("function"), dict):
        call = call["function"]
    name = call.get("name") if isinstance(call, dict) else None
    args = (call.get("arguments") or call.get("parameters") or {}) if isinstance(call, dict) else {}
    if name not in known or not isinstance(args, dict):
        return result
    return {**result, "content": "", "tool_calls": [
        {"id": "call_" + uuid.uuid4().hex[:8], "function": {"name": name, "arguments": args}}]}


def tool_call_to_action_data(tool_call: dict) -> dict:
    """
    Adapt a native tool call {name, arguments} into the action_data dict
    shape the rest of cortex.py already understands.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    args = fn.get("arguments") or {}

    if name == "execute_shell":
        return {"action": "execute", "command": args.get("command"), "background": bool(args.get("background"))}
    if name == "web_search":
        return {"action": "search", "search_query": args.get("query")}
    if name == "browse_url":
        return {"action": "browse", "browse_url": args.get("url"), "browse_open": bool(args.get("open_in_browser", False))}
    if name == "browser_automate":
        return {"action": "browser", "browser_url": args.get("url"), "browser_steps": args.get("steps") or []}
    if name == "crawl_site":
        return {
            "action": "crawl", "crawl_url": args.get("url"),
            "crawl_credential": args.get("credential"), "crawl_max_pages": args.get("max_pages", 3),
        }
    if name == "edit_file":
        return {"action": "edit", "edit_file": args.get("path"), "edit_instruction": args.get("instruction")}
    if name == "run_project":
        return {"action": "run_project", "run_project_path": args.get("path"), "run_project_command": args.get("command")}
    if name == "make_chart":
        return {"action": "chart", "code": args.get("code"), "caption": args.get("caption", "")}
    if name == "image_search":
        return {"action": "image_search", "query": args.get("query")}
    if name == "ask_user":
        return {"action": "ask", "question": args.get("question"), "options": args.get("options") or []}
    if name == "email_check":
        return {"action": "email_check"}
    if name == "email_search":
        return {"action": "email_search", "email_query": args.get("query")}
    if name == "email_send":
        return {"action": "email_send", "email_to": args.get("to"), "email_subject": args.get("subject"),
                "email_body": args.get("body"), "email_from": args.get("from")}
    if name == "manage_reminder":
        return {"action": "remind", "reminder": {
            "op": args.get("op", "create"), "text": args.get("text"),
            "due": args.get("due"), "repeat": args.get("repeat"), "id": args.get("id"),
        }}
    if name == "schedule_task":
        return {"action": "schedule_task", "task_op": args.get("op", "add"),
                "task_desc": args.get("desc"), "task_prompt": args.get("prompt"),
                "task_schedule": args.get("schedule"), "task_deliver": args.get("deliver") or "notify",
                "task_id": args.get("id")}
    if name == "remember":
        return {"action": "remember", "remember_content": args.get("content"),
                "remember_type": args.get("type", "fact")}
    if name == "forget":
        return {"action": "forget", "forget_id": args.get("memory_id"),
                "forget_search": args.get("search")}
    if name == "delegate":
        return {"action": "delegate", "subtasks": args.get("subtasks") or []}
    if name == "manage_workers":
        return {"action": "workers", "workers_op": args.get("op", "list"),
                "worker": args.get("worker"), "profile": args.get("profile")}
    if name == "watch_process":
        return {"action": "watch", "watch_op": args.get("op", "add"), "watch_label": args.get("label"),
                "watch_note": args.get("note"), "watch_pid": args.get("pid"), "watch_process": args.get("process_name"),
                "watch_port": args.get("port"), "watch_file": args.get("file"), "watch_pattern": args.get("pattern"),
                "watch_command": args.get("check_command"), "watch_until": args.get("until"),
                "watch_id": args.get("id")}
    if name == "undo_changes":
        return {"action": "undo"}
    if name == "request_credentials":
        return {"action": "credential_prompt", "credential_prompt_site": args.get("site"), "credential_prompt_for": args.get("purpose")}

    return {"action": "answer"}
