"""
Cortex — The Control Layer
Main daemon: Flask HTTP server.
Always-on via launchd. Receives commands from Siri/terminal/API.
Wired: DB, LLM, Executor, Context, Memory, Reminders.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime

import yaml
from flask import Flask, request, jsonify

from db import Database
from llm import create_router
from executor import CommandExecutor
from context_reader import ContextReader
from memory import MemoryManager
from reminders import ReminderManager
from devops_agent import DevOpsAgent
from multi_scope import is_multi_scope, execute_multi_scope

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("cortex")

# ─── Load Config ───
CONFIG_PATHS = [
    os.path.join(os.path.dirname(__file__), "config.yaml"),
    os.path.expanduser("~/.cortex/config.yaml"),
]


def load_config() -> dict:
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            logger.info(f"Config loaded from {path}")
            return cfg
    logger.error("No config.yaml found!")
    sys.exit(1)


config = load_config()

# ─── Initialize Components ───
db = Database(
    host=config["database"]["host"],
    port=config["database"]["port"],
    name=config["database"]["name"],
    user=config["database"]["user"],
    password=config["database"].get("password") or os.environ.get("CORTEX_DB_PASSWORD", "")
)

if not db.connect():
    logger.error("Cannot start without database connection. Exiting.")
    sys.exit(1)

llm_router = create_router(config["llm"])
executor = CommandExecutor(db)
context_reader = ContextReader(db, config.get("context", {}))
memory_mgr = MemoryManager(db, llm_router, config.get("memory", {}))
reminder_mgr = ReminderManager(db, config.get("reminders", {}))
devops_agent = DevOpsAgent(db, llm_router, executor, context_reader)

# Start background reminder checker
reminder_mgr.start()

# ─── Flask App ───
app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════
# MAIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.route("/ask", methods=["POST"])
def ask():
    """
    Main intelligence endpoint — the brain.
    Accepts text, returns spoken response + optional action.
    This is what Siri calls.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    source = data.get("source", "api")
    session_id = data.get("session_id", f"{source}-{datetime.now().strftime('%Y%m%d')}")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Save user message
    db.save_conversation(session_id, "user", text)

    # Get memory context
    memory_str = memory_mgr.format_for_prompt(limit=15)

    # Check if this is a multi-scope query (needs parallel data gathering)
    extra_context = ""
    if is_multi_scope(text):
        tracked = db.get_tracked_projects()
        extra_context = execute_multi_scope(text, tracked_projects=tracked)

    # Build messages with context + memory + multi-scope data
    messages = context_reader.build_messages(
        user_text=text,
        session_id=session_id,
        project_path=data.get("project_path"),
        memory_str=memory_str,
        extra_context=extra_context,
        source=source
    )

    # Call LLM
    try:
        response_text = llm_router.complete(messages, json_mode=True)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return jsonify({
            "error": "LLM unavailable",
            "speak": "Sorry, I can't reach my brain right now."
        }), 503

    # Parse response
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        response_json = {"speak": response_text, "action": "answer", "command": None}

    # Save assistant response (include execution output for follow-up context)
    save_content = json.dumps(response_json)
    if response_json.get("execution_result"):
        # Append a summary of what was found so follow-up questions have context
        exec_summary = (
            f"\n[Executed: {response_json.get('command', '')}]\n"
            f"[Output: {response_json['execution_result'].get('stdout', '')[:800]}]"
        )
        db.save_conversation(session_id, "assistant",
                             response_json.get("speak", "") + exec_summary)
    else:
        db.save_conversation(session_id, "assistant", response_json.get("speak", save_content))

    # ─── Action handling ───

    # Execute command
    if response_json.get("action") == "execute" and response_json.get("command"):
        cmd_result = executor.execute(
            command=response_json["command"],
            cwd=data.get("project_path"),
            source=source
        )
        response_json["execution_result"] = cmd_result

        # Summarize the command output into a useful spoken response
        if cmd_result.get("stdout") or cmd_result.get("stderr"):
            try:
                summary_messages = [
                    {"role": "system", "content": (
                        "You are Cortex, a developer's AI assistant. You just ran a command for the user.\n\n"
                        "Your job is to present the results clearly in markdown. Rules:\n"
                        "- If the output is a LIST (ports, files, containers, branches) → INCLUDE THE ACTUAL DATA formatted as a markdown list or table. Don't just say 'here are the ports' without listing them.\n"
                        "- If the output is complex/verbose → summarize what matters, highlight concerns, suggest actions.\n"
                        "- If there's an error → explain what went wrong and how to fix it.\n"
                        "- Use markdown formatting: **bold** for emphasis, `code` for commands/names, bullet lists for items.\n"
                        "- Be conversational but ALWAYS include the actual information the user asked for.\n\n"
                        "Respond with JSON: {\"speak\": \"your markdown-formatted response with actual data\"}"
                    )},
                    {"role": "user", "content": (
                        f"User asked: {text}\n"
                        f"Command ran: {response_json['command']}\n"
                        f"Exit code: {cmd_result['exit_code']}\n"
                        f"Output:\n{cmd_result['stdout'][:2000]}\n"
                        f"Errors:\n{cmd_result['stderr'][:500]}"
                    )}
                ]
                summary_raw = llm_router.complete(summary_messages, json_mode=True)
                summary = json.loads(summary_raw)
                if summary.get("speak"):
                    # Safety check: if summary is suspiciously short/empty compared to output,
                    # append raw output as a fallback
                    speak = summary["speak"]
                    stdout = cmd_result.get("stdout", "").strip()
                    if len(speak) < 80 and len(stdout) > 50:
                        # Summary is too short — likely dropped the data. Append it.
                        speak += "\n\n```\n" + stdout[:1500] + "\n```"
                    response_json["speak"] = speak
            except Exception as e:
                logger.warning(f"Output summarization failed: {e}")

    # Create reminder
    if response_json.get("action") == "remind" and response_json.get("reminder"):
        reminder = response_json["reminder"]
        try:
            due = datetime.fromisoformat(reminder["due"])
            reminder_id = db.create_reminder(
                text=reminder["text"],
                due_at=due,
                repeat_rule=reminder.get("repeat")
            )
            response_json["reminder_id"] = reminder_id
        except Exception as e:
            logger.error(f"Failed to create reminder: {e}")

    # ─── Memory extraction (async-ish, after response) ───
    memory_mgr.on_conversation_turn(session_id)

    return jsonify(response_json)


@app.route("/execute", methods=["POST"])
def execute():
    """Direct command execution — raw shell access."""
    data = request.get_json(force=True)
    command = data.get("command", "").strip()

    if not command:
        return jsonify({"error": "No command provided"}), 400

    result = executor.execute(
        command=command,
        cwd=data.get("cwd"),
        timeout=data.get("timeout", 60),
        source=data.get("source", "api")
    )
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════
# REMINDERS
# ═══════════════════════════════════════════════════════════════

@app.route("/remind", methods=["POST"])
def remind():
    """Set a reminder via natural language."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    parse_messages = [
        {"role": "system", "content": (
            "Extract a reminder from the user's text. Respond with JSON only:\n"
            '{"text": "what to remind", "due": "ISO 8601 timestamp", "repeat": null or "daily"|"weekly"|"weekdays"}\n'
            f"Current time: {datetime.now().isoformat()}"
        )},
        {"role": "user", "content": text}
    ]

    try:
        parsed = llm_router.complete(parse_messages, json_mode=True)
        reminder_data = json.loads(parsed)
        due = datetime.fromisoformat(reminder_data["due"])
        reminder_id = db.create_reminder(
            text=reminder_data["text"],
            due_at=due,
            repeat_rule=reminder_data.get("repeat")
        )
        return jsonify({
            "speak": f"Got it. I'll remind you: {reminder_data['text']}",
            "reminder_id": reminder_id,
            "due": reminder_data["due"]
        })
    except Exception as e:
        logger.error(f"Reminder parsing failed: {e}")
        return jsonify({
            "error": "Could not parse reminder",
            "speak": "I couldn't understand the timing."
        }), 400


@app.route("/reminders", methods=["GET"])
def list_reminders():
    """List all active (unfired) reminders."""
    active = reminder_mgr.get_active()
    formatted = []
    for r in active:
        formatted.append({
            "id": r["id"],
            "text": r["text"],
            "due_at": r["due_at"].isoformat() if hasattr(r["due_at"], "isoformat") else str(r["due_at"]),
            "repeat": r.get("repeat_rule"),
        })
    return jsonify({"reminders": formatted, "count": len(formatted)})


# ═══════════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════════

@app.route("/memory", methods=["GET"])
def get_memory():
    """Query stored memories."""
    mem_type = request.args.get("type")
    limit = int(request.args.get("limit", 20))
    memories = db.get_memories(type=mem_type, limit=limit)
    return jsonify({"memories": memories, "count": len(memories)})


@app.route("/memory", methods=["POST"])
def add_memory():
    """Manually add a memory fact."""
    data = request.get_json(force=True)
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "No content provided"}), 400

    memory_mgr.store_fact(
        content=content,
        mem_type=data.get("type", "fact"),
        source=data.get("source", "manual"),
        importance=int(data.get("importance", 4))
    )
    return jsonify({"speak": f"I'll remember that.", "stored": content})


@app.route("/memory/extract", methods=["POST"])
def extract_memory():
    """Force memory extraction from a session's conversation."""
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    memory_mgr._extract_facts(session_id)
    return jsonify({"speak": "Memory extraction complete.", "session_id": session_id})


# ═══════════════════════════════════════════════════════════════
# DEVOPS
# ═══════════════════════════════════════════════════════════════

@app.route("/devops", methods=["POST"])
def devops():
    """Multi-step DevOps task — returns a plan for approval."""
    data = request.get_json(force=True)
    task = data.get("task", "").strip()
    project_path = data.get("project")
    auto_execute = data.get("auto_execute", False)

    if not task:
        return jsonify({"error": "No task provided"}), 400

    result = devops_agent.create_plan(task=task, project_path=project_path)

    if auto_execute and result.get("plan"):
        return jsonify(devops_agent.execute_plan(result["plan_id"]))

    return jsonify(result)


@app.route("/devops/execute", methods=["POST"])
def devops_execute():
    """Execute a previously approved plan."""
    data = request.get_json(force=True)
    plan_id = data.get("plan_id", "").strip()
    if not plan_id:
        return jsonify({"error": "No plan_id provided"}), 400

    result = devops_agent.execute_plan(plan_id)
    return jsonify(result)


def _execute_plan(plan_id: str):
    """Internal: kept for compatibility, delegates to agent."""
    return jsonify(devops_agent.execute_plan(plan_id))


# ═══════════════════════════════════════════════════════════════
# STATUS & PROJECTS
# ═══════════════════════════════════════════════════════════════

@app.route("/status", methods=["GET"])
def status():
    """Daemon health."""
    db_ok = db.health_check()
    projects = db.get_tracked_projects() if db_ok else []
    active_reminders = reminder_mgr.get_active() if db_ok else []
    memories = db.get_memories(limit=1) if db_ok else []

    return jsonify({
        "daemon": "running",
        "db_connected": db_ok,
        "llm_primary": config["llm"].get("primary", "openai"),
        "llm_fallback": config["llm"].get("fallback"),
        "tracked_projects": len(projects),
        "active_reminders": len(active_reminders),
        "memories_stored": len(memories) > 0,
        "server_port": config["server"]["port"],
    })


@app.route("/projects/track", methods=["POST"])
def track_project():
    """Register a project for context collection."""
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    name = data.get("name", "").strip()
    if not path or not name:
        return jsonify({"error": "path and name required"}), 400

    db.track_project(
        name=name, path=path,
        project_type=data.get("type"),
        custom_commands=data.get("custom_commands")
    )
    return jsonify({"speak": f"Now tracking {name}.", "path": path})


# ═══════════════════════════════════════════════════════════════
# CONVERSATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@app.route("/conversations", methods=["GET"])
def get_conversations():
    """Get conversation history for a session."""
    session_id = request.args.get("session_id", "")
    limit = int(request.args.get("limit", 20))
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    history = db.get_conversations(session_id=session_id, limit=limit)
    return jsonify({"session_id": session_id, "messages": history})


@app.route("/reload", methods=["POST"])
def reload_daemon():
    """Hot-reload: restart daemon process (useful during dev)."""
    import os
    import sys
    logger.info("Reload requested — restarting process...")
    # Restart by re-execing the same process
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ─── Entry Point ───
if __name__ == "__main__":
    port = config["server"]["port"]
    host = config["server"]["host"]
    logger.info(f"Cortex daemon starting on {host}:{port}")
    logger.info(f"Memory: max_facts={memory_mgr.max_facts}, extract_every={memory_mgr.summarize_every} turns")
    logger.info(f"Reminders: checking every {reminder_mgr.check_interval}s")
    app.run(host=host, port=port, debug=False)
