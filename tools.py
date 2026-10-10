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
            "description": "Modify a file's contents. Always held for user confirmation before it actually runs.",
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
            "description": "Detect a tracked project's type and run its dev/start/build/test/install command.",
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


def tool_call_to_action_data(tool_call: dict) -> dict:
    """
    Adapt a native tool call {name, arguments} into the action_data dict
    shape the rest of cortex.py already understands.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    args = fn.get("arguments") or {}

    if name == "execute_shell":
        return {"action": "execute", "command": args.get("command")}
    if name == "web_search":
        return {"action": "search", "search_query": args.get("query")}
    if name == "browse_url":
        return {"action": "browse", "browse_url": args.get("url"), "browse_open": bool(args.get("open_in_browser", False))}
    if name == "crawl_site":
        return {
            "action": "crawl", "crawl_url": args.get("url"),
            "crawl_credential": args.get("credential"), "crawl_max_pages": args.get("max_pages", 3),
        }
    if name == "edit_file":
        return {"action": "edit", "edit_file": args.get("path"), "edit_instruction": args.get("instruction")}
    if name == "run_project":
        return {"action": "run_project", "run_project_path": args.get("path"), "run_project_command": args.get("command")}
    if name == "manage_reminder":
        return {"action": "remind", "reminder": {
            "op": args.get("op", "create"), "text": args.get("text"),
            "due": args.get("due"), "repeat": args.get("repeat"), "id": args.get("id"),
        }}
    if name == "request_credentials":
        return {"action": "credential_prompt", "credential_prompt_site": args.get("site"), "credential_prompt_for": args.get("purpose")}

    return {"action": "answer"}
