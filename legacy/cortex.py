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
import threading
import subprocess
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import yaml
from flask import Flask, request, jsonify, Response, stream_with_context

from db import Database
from llm import create_router
from executor import CommandExecutor
from context_reader import ContextReader
from memory import MemoryManager
from reminders import ReminderManager
from devops_agent import DevOpsAgent
from multi_scope import is_multi_scope, execute_multi_scope
from appflowy import create_client as create_appflowy_client
from websearch import search_and_format
from webbrowser import open_in_browser, fetch_and_format as browse_and_format
from webcrawler import crawl_and_format, crawl_with_credentials
from credential_prompt import prompt_credentials, prompt_single_input, prompt_choice
from file_editor import read_file as editor_read_file, write_file as editor_write_file, apply_edit, apply_patch, detect_project, find_file, list_directory
from vision import analyze_screen, analyze_screen_for_voice
from calendar_integration import get_todays_events, get_upcoming_events, format_events_for_context
from clipboard_watch import ClipboardWatcher
from automation import AutomationEngine
from scheduled_reports import ReportScheduler
from multi_machine import RemoteExecutor, get_tailscale_peers
from safety import needs_confirmation
from autonomy import AutonomyEngine
from schemas import SPEAK_SCHEMA, EDIT_SCHEMA, REMINDER_PARSE_SCHEMA
from tools import TOOLS, tool_call_to_action_data

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

# AppFlowy (optional — only if configured)
appflowy_client = None
if config.get("appflowy", {}).get("email") or config.get("appflowy", {}).get("token"):
    appflowy_client = create_appflowy_client(config)
    logger.info("AppFlowy client initialized")

# Start background reminder checker
reminder_mgr.start()

# Automation engine
automation_engine = AutomationEngine(config.get("automation_rules", []), db=db)
automation_engine.start()

# Scheduled reports
report_scheduler = ReportScheduler(config)
report_scheduler.start()

# Clipboard watcher
clipboard_watcher = ClipboardWatcher()
clipboard_watcher.start()

# Multi-machine remote executor
remote_executor = RemoteExecutor(config.get("remote_hosts", {}))

# ─── Web Credentials ───
web_credentials = config.get("web_credentials", {})


def get_credentials_list() -> str:
    """Build a human-readable list of available credential names for the LLM prompt."""
    if not web_credentials:
        return "(No web credentials configured — user can still provide URLs for unauthenticated browsing)"
    lines = []
    for name, cred in web_credentials.items():
        auth_type = cred.get("auth_type", "unknown")
        lines.append(f"  • {name} ({auth_type})")
    return "\n".join(lines)


def resolve_credential(name: str) -> dict:
    """
    Resolve a credential by name, filling in env vars for passwords/tokens.
    Convention: CORTEX_CRED_{NAME}_{FIELD} e.g. CORTEX_CRED_JIRA_PASSWORD
    """
    cred = web_credentials.get(name)
    if not cred:
        return None

    resolved = dict(cred)
    upper_name = name.upper()

    # Resolve password from env if empty
    if not resolved.get("password"):
        env_key = f"CORTEX_CRED_{upper_name}_PASSWORD"
        resolved["password"] = os.environ.get(env_key, resolved.get("password", ""))

    # Resolve token from env if empty
    if not resolved.get("token"):
        env_key = f"CORTEX_CRED_{upper_name}_TOKEN"
        resolved["token"] = os.environ.get(env_key, resolved.get("token", ""))

    # Resolve headers from env if empty values
    if resolved.get("headers"):
        for header_key, header_val in resolved["headers"].items():
            if not header_val:
                env_key = f"CORTEX_CRED_{upper_name}_{header_key.upper().replace('-', '_')}"
                resolved["headers"][header_key] = os.environ.get(env_key, "")

    return resolved


def get_projects_list() -> str:
    """Build a human-readable list of tracked projects for the LLM prompt."""
    projects = config.get("projects", [])
    if not projects:
        return "(No tracked projects — user can still provide paths)"
    lines = []
    for proj in projects:
        name = proj.get("name", "unnamed")
        path = proj.get("path", "?")
        lines.append(f"  • {name} → {path}")
    return "\n".join(lines)


# ─── Flask App ───
app = Flask(__name__)

# Pending actions waiting for user confirmation (session_id → action_data)
_pending_actions = {}

# Most recent experience id per session (for attaching outcomes after the fact)
_last_experience = {}

# Guards read-modify-write sequences on the two dicts above — now that the
# server runs threaded, concurrent requests for different sessions can
# interleave on them.
_state_lock = threading.Lock()

# Background work that shouldn't sit on the response path: experience
# logging (does its own embedding call), and anything similar.
_bg = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cortex-bg")

# Heuristic phrases that signal the PREVIOUS turn's action was wrong / needed correction
_CORRECTION_PHRASES = [
    "no i meant", "that's not what", "not what i asked", "wrong", "that's wrong",
    "i said", "i meant", "no, i", "not that", "try again", "that's not it",
    "i didn't ask", "you misunderstood", "that's incorrect", "no that's not",
]


def _result_to_response_json(result: dict) -> dict:
    """
    Adapt a complete_with_tools() result into the action_data dict shape the
    rest of ask() (confirmation gate, per-action branches, experience log)
    already understands. Only the first tool call is used — one action per
    turn, matching the existing design. No tool call = a direct answer,
    which is also how the model expresses STANDING PRIORITIES pushback now
    (no JSON gymnastics needed for that case).
    """
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        response_json = tool_call_to_action_data(tool_calls[0])
        response_json["think"] = result.get("thinking", "")
        # Every per-action branch downstream overwrites "speak" with a real
        # summary once it actually runs (confirmed for execute/search/browse/
        # crawl/remind/run_project/credential_prompt) — this placeholder is
        # only ever user-visible for the confirmation-prompt case, which
        # comes from needs_confirmation()'s own message instead.
        response_json["speak"] = result.get("content") or "One moment..."
        return response_json
    return {
        "action": "answer",
        "command": None,
        "think": result.get("thinking", ""),
        "speak": result.get("content") or "I'm not sure how to respond to that.",
    }


def _log_experience(session_id: str, user_text: str, response_json: dict):
    """Log this turn as an experience (State-Action-Reward buffer, unscored for now)."""
    try:
        exp_id = db.log_experience(
            session_id=session_id,
            user_text=user_text,
            action=response_json.get("action", "answer"),
            action_payload=response_json,
            speak=response_json.get("speak"),
        )
        with _state_lock:
            _last_experience[session_id] = exp_id
        return exp_id
    except Exception as e:
        logger.warning(f"Experience logging failed: {e}")
        return None


def _log_experience_bg(session_id: str, user_text: str, response_json: dict):
    """
    Background version of the hot-path experience log: does the embedding
    call, insert, and (if there's an exec result) the outcome update as one
    job on _bg, so none of it sits on the /ask response path.
    """
    exp_id = _log_experience(session_id, user_text, response_json)
    exec_result = response_json.get("execution_result")
    if exp_id and exec_result and exec_result.get("exit_code") is not None:
        try:
            db.update_experience_outcome(exp_id, exit_code=exec_result["exit_code"])
        except Exception as e:
            logger.warning(f"Background outcome update failed: {e}")


def _execute_held_action(held: dict, source: str) -> tuple:
    """
    Run a previously-confirmed action (edit / remind-delete / execute-devops).
    Shared by the /ask confirmation flow and the /autonomy proposal-confirm
    endpoint so there's one place that knows how to actually carry out a
    held action. Returns (speak, cmd_result_or_None).
    """
    action = held.get("action", "")
    speak = held.get("speak", "Done.")
    cmd_result = None

    if action == "edit" and held.get("edit_file"):
        edit_filepath = held["edit_file"]
        edit_instruction = held.get("edit_instruction", "")
        from file_editor import read_file as editor_read_file, apply_edit
        file_data = editor_read_file(edit_filepath)
        if file_data["success"]:
            try:
                edit_messages = [
                    {"role": "system", "content": (
                        "Given a file and instruction, produce the edit.\n"
                        "Respond with JSON: {\"find\": \"exact text\", \"replace\": \"new text\", \"explanation\": \"what changed\"}"
                    )},
                    {"role": "user", "content": f"FILE: {edit_filepath}\nINSTRUCTION: {edit_instruction}\n\nCONTENT:\n{file_data['content'][:15000]}"}
                ]
                edit_raw = llm_router.complete(edit_messages, json_mode=True, schema=EDIT_SCHEMA)
                edit_data = json.loads(edit_raw)
                result = apply_edit(edit_filepath, edit_data["find"], edit_data["replace"])
                speak = f"Done. {edit_data.get('explanation', '')}" if result["success"] else f"Edit failed: {result.get('error')}"
            except Exception as e:
                speak = f"Edit failed: {e}"
        else:
            speak = f"Can't read {edit_filepath}: {file_data['error']}"

    elif action == "remind" and held.get("reminder", {}).get("op") == "delete":
        rid = held.get("reminder", {}).get("id")
        success = db.delete_reminder(int(rid))
        speak = f"Deleted reminder #{rid}." if success else f"No reminder #{rid} found."

    elif action in ("execute", "devops") and held.get("command"):
        cmd_result = executor.execute(command=held["command"], source=source)
        speak = f"Done. Exit code: {cmd_result['exit_code']}"
        if cmd_result.get("stdout"):
            speak += f"\n```\n{cmd_result['stdout'][:1000]}\n```"

    return speak, cmd_result


def _check_for_correction(session_id: str, text: str):
    """
    If this new message looks like the user correcting/rephrasing the previous
    request, mark the PREVIOUS experience as corrected (negative reward signal).
    This is the "implicit feedback" half of the reward function — no explicit
    thumbs up/down needed, just noticing when the user had to push back.
    """
    text_lower = text.lower()
    if not any(p in text_lower for p in _CORRECTION_PHRASES):
        return
    with _state_lock:
        prev_id = _last_experience.get(session_id)
    if prev_id:
        try:
            db.update_experience_outcome(prev_id, was_corrected=True)
            logger.info(f"Experience {prev_id} marked as corrected (user pushback detected)")
        except Exception as e:
            logger.warning(f"Failed to mark correction: {e}")


def _reward_loop():
    """Background thread: periodically score experiences that have outcome signals."""
    import time as _time
    while True:
        try:
            db.compute_pending_rewards()
        except Exception as e:
            logger.warning(f"Reward computation failed: {e}")
        _time.sleep(60)


# BIG BRAIN System 2 substitute: background reward scoring over the experience buffer.
# No weight updates happen (impossible against a closed-weight model like GPT-4o) —
# this just scores outcomes so retrieval in context_reader.py can prefer what worked.
threading.Thread(target=_reward_loop, daemon=True, name="reward-loop").start()

# BIG BRAIN autonomy loop: goal-directed background cognition, gated through
# the same needs_confirmation() safety check /ask uses for live requests.
autonomy_engine = AutonomyEngine(db, llm_router, executor, context_reader, memory_mgr, config.get("autonomy", {}))
autonomy_engine.start()


# ═══════════════════════════════════════════════════════════════
# MAIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def stream_response(messages):
    """Generator that yields LLM response chunks as SSE stream."""
    full_text = ""
    try:
        for chunk in llm_router.stream(messages):
            full_text += chunk
            # Yield as Server-Sent Events format
            data = json.dumps({"chunk": chunk})
            yield f"data: {data}\n\n"
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
    
    # Send final done marker with the full accumulated text
    done_data = json.dumps({"done": True, "full_text": full_text})
    yield f"data: {done_data}\n\n"


@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    """
    Streaming version of /ask endpoint. Returns Server-Sent Events (SSE)
    with incremental LLM response chunks for real-time display.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    session_id = data.get("session_id", f"cli-{datetime.now().strftime('%Y%m%d-%H%M')}")
    source = data.get("source", "cli")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Save user message
    db.save_conversation(session_id, "user", text)

    # Get memory context
    memory_str = memory_mgr.format_for_prompt(limit=15)

    # Build messages (simplified — no multi-scope for streaming yet)
    messages = context_reader.build_messages(
        user_text=text,
        session_id=session_id,
        project_path=data.get("project_path"),
        memory_str=memory_str,
        extra_context="",
        source=source,
        credentials_list=get_credentials_list(),
        projects_list=get_projects_list()
    )

    # Return SSE stream
    response = Response(
        stream_with_context(stream_response(messages)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
    return response


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

    # Implicit feedback: does this message look like a correction of the last turn?
    _check_for_correction(session_id, text)

    # Check if user is responding to a pending report
    if report_scheduler.has_pending():
        text_lower = text.lower().strip()
        affirm = ["yes", "yeah", "go ahead", "tell me", "sure", "yep", "ok", "ready", "shoot"]
        defer = ["not now", "later", "no", "nope", "busy", "in a bit"]

        if any(word in text_lower for word in affirm):
            content = report_scheduler.deliver_report()
            db.save_conversation(session_id, "user", text)
            db.save_conversation(session_id, "assistant", content)
            return jsonify({"speak": content, "action": "answer", "command": None})

        elif any(word in text_lower for word in defer):
            # Try to extract time from response
            minutes = 30
            if "10" in text_lower:
                minutes = 10
            elif "15" in text_lower:
                minutes = 15
            elif "30" in text_lower or "half" in text_lower:
                minutes = 30
            elif "hour" in text_lower or "60" in text_lower:
                minutes = 60

            result = report_scheduler.reschedule(minutes=minutes)
            return jsonify({"speak": result, "action": "answer", "command": None})

    # ─── Pending action confirmation ───
    # If the agent previously asked for confirmation, check if user is responding
    pending_key = f"pending_action:{session_id}"
    with _state_lock:
        pending_action = _pending_actions.get(pending_key)

    if pending_action:
        text_lower_resp = text.lower().strip()
        affirm = ["yes", "yeah", "go ahead", "sure", "yep", "ok", "do it", "confirm", "proceed", "y"]
        deny = ["no", "nope", "cancel", "don't", "never mind", "stop", "n"]

        if any(word == text_lower_resp or word in text_lower_resp.split() for word in affirm):
            # Execute the held action immediately
            with _state_lock:
                _pending_actions.pop(pending_key, None)
            db.save_conversation(session_id, "user", text)

            held = pending_action["action_data"]
            held.pop("_confirmed", None)
            logger.info(f"Executing confirmed action: {held.get('action')}")

            speak, cmd_result = _execute_held_action(held, source)
            result = {"speak": speak, "action": "answer"}
            db.save_conversation(session_id, "assistant", speak)

            # Log this confirmed action as an experience with its outcome already known
            exp_id = _log_experience(session_id, text, held)
            if exp_id:
                if cmd_result:
                    db.update_experience_outcome(exp_id, exit_code=cmd_result.get("exit_code"), was_confirmed=True)
                else:
                    db.update_experience_outcome(exp_id, was_confirmed=True)

            memory_mgr.on_conversation_turn(session_id)
            return jsonify(result)

        elif any(word == text_lower_resp or word in text_lower_resp.split() for word in deny):
            with _state_lock:
                _pending_actions.pop(pending_key, None)
            db.save_conversation(session_id, "user", text)
            response_json = {"speak": "Alright, cancelled.", "action": "answer"}
            db.save_conversation(session_id, "assistant", response_json["speak"])

            # Rejected action is a real negative-reward signal — log it
            exp_id = _log_experience(session_id, text, pending_action["action_data"])
            if exp_id:
                db.update_experience_outcome(exp_id, was_confirmed=False)

            memory_mgr.on_conversation_turn(session_id)
            return jsonify(response_json)

        else:
            # User said something else — clear pending and process normally
            with _state_lock:
                _pending_actions.pop(pending_key, None)

    # Save user message
    db.save_conversation(session_id, "user", text)

    # ─── Intercept: Briefing/status requests → generate report directly ───
    briefing_keywords = ["briefing", "morning briefing", "today's briefing", "status update",
                         "catch me up", "what's going on", "system status", "give me a rundown",
                         "daily report", "evening report", "what did i miss"]
    text_lower_check = text.lower().strip()
    if any(kw in text_lower_check for kw in briefing_keywords):
        # Generate a live briefing by running the morning briefing prompt through cortex itself
        briefing_prompt = (
            "Give me a quick briefing. Check: system health (CPU, disk, memory), "
            "Docker container status (any crashed/exited?), recent git activity across my projects, "
            "and my calendar for today. Run the actual commands and report real data."
        )
        # Build messages with briefing prompt instead of user text
        briefing_memory = memory_mgr.format_for_prompt(limit=15)
        briefing_messages = context_reader.build_messages(
            user_text=briefing_prompt,
            session_id=session_id,
            project_path=data.get("project_path"),
            memory_str=briefing_memory,
            source=source,
            credentials_list=get_credentials_list(),
            projects_list=get_projects_list()
        )
        try:
            result = llm_router.complete_with_tools(briefing_messages, tools=TOOLS)
            response_json = _result_to_response_json(result)
        except Exception:
            response_json = {"speak": "Couldn't generate the briefing right now.", "action": "answer"}

        # If it wants to execute a command, let it
        if response_json.get("action") in ("execute", "devops") and response_json.get("command"):
            cmd_result = executor.execute(
                command=response_json["command"],
                cwd=data.get("project_path"),
                source=source
            )
            # Summarize
            try:
                summary_messages = [
                    {"role": "system", "content": (
                        "You are Cortex. You ran system commands for a daily briefing.\n"
                        "Summarize what you found into a concise briefing. Include:\n"
                        "- System health (CPU, disk, memory)\n"
                        "- Container status (running/crashed)\n"
                        "- Any issues or concerns\n"
                        "Keep it conversational and short for voice.\n"
                        "Respond with JSON: {\"speak\": \"your briefing\"}"
                    )},
                    {"role": "user", "content": (
                        f"Commands ran: {response_json['command']}\n"
                        f"Output:\n{cmd_result['stdout'][:3000]}\n"
                        f"Errors:\n{cmd_result['stderr'][:500]}"
                    )}
                ]
                raw = llm_router.complete(summary_messages, json_mode=True, schema=SPEAK_SCHEMA)
                parsed = json.loads(raw)
                response_json["speak"] = parsed.get("speak", response_json.get("speak", ""))
            except Exception:
                response_json["speak"] = f"Here's what I found:\n\n{cmd_result['stdout'][:1500]}"

        db.save_conversation(session_id, "assistant", response_json.get("speak", ""))
        memory_mgr.on_conversation_turn(session_id)
        return jsonify(response_json)

    # ─── Intercept: News requests → always web search, never "you can check..." ───
    news_keywords = ["latest news", "today's news", "headlines", "top stories",
                     "current events", "what happened today", "news update",
                     "tell me the news", "get me news", "bbc news", "world news"]
    if any(kw in text_lower_check for kw in news_keywords):
        # Force a web search and synthesize
        # search_and_format is already imported at top level
        search_query = text  # Use their exact query as the search
        logger.info(f"News intercept: searching for '{search_query}'")
        search_results = search_and_format(search_query, max_results=8)

        try:
            if source == "siri":
                synth_prompt = (
                    "You searched the web for news. Give a concise spoken summary "
                    "of the top findings in 2-4 sentences. Natural speech, no formatting.\n"
                    "Respond with JSON: {\"speak\": \"your summary\"}"
                )
            else:
                synth_prompt = (
                    "You searched the web for news. Present the key findings clearly.\n"
                    "Respond with JSON: {\"speak\": \"your markdown summary\"}"
                )

            search_messages = [
                {"role": "system", "content": synth_prompt},
                {"role": "user", "content": f"User asked: {text}\n\nSearch results:\n{search_results}"}
            ]
            raw = llm_router.complete(search_messages, json_mode=True, schema=SPEAK_SCHEMA)
            parsed = json.loads(raw)
            response_json = {"speak": parsed.get("speak", ""), "action": "answer"}
        except Exception:
            response_json = {"speak": f"Here's what I found:\n\n{search_results[:2000]}", "action": "answer"}

        db.save_conversation(session_id, "assistant", response_json.get("speak", ""))
        memory_mgr.on_conversation_turn(session_id)
        return jsonify(response_json)

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
        source=source,
        credentials_list=get_credentials_list(),
        projects_list=get_projects_list()
    )

    # Call LLM — native tool-calling: the model either calls one of TOOLS or
    # just answers in plain text (which is also how it expresses STANDING
    # PRIORITIES pushback). See tools.py for why this replaced a JSON envelope.
    try:
        result = llm_router.complete_with_tools(messages, tools=TOOLS)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return jsonify({
            "error": "LLM unavailable",
            "speak": "Sorry, I can't reach my brain right now."
        }), 503

    response_json = _result_to_response_json(result)

    # Log the model's reasoning (useful for debugging)
    if response_json.get("think"):
        logger.info(f"LLM reasoning: {response_json['think'][:200]}")

    # ─── Confirmation gate for destructive actions ───
    # If the action is destructive and wasn't already confirmed, hold it and ask.
    # Shared with the autonomy loop (safety.py) so live requests and self-directed
    # actions are never held to different bars.
    if not response_json.get("_confirmed"):
        needs_confirm, confirm_msg = needs_confirmation(response_json)

        if needs_confirm:
            pending_key = f"pending_action:{session_id}"
            with _state_lock:
                _pending_actions[pending_key] = {
                    "action_data": response_json,
                    "asked_at": datetime.now().isoformat(),
                }
            # Return the confirmation question instead of executing
            confirm_response = {"speak": confirm_msg, "action": "answer", "awaiting_confirmation": True}
            db.save_conversation(session_id, "assistant", confirm_msg)
            memory_mgr.on_conversation_turn(session_id)
            return jsonify(confirm_response)

    # Clear the confirmed flag before proceeding
    response_json.pop("_confirmed", None)

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
    if response_json.get("action") in ("execute", "devops") and response_json.get("command"):
        cmd_result = executor.execute(
            command=response_json["command"],
            cwd=data.get("project_path"),
            source=source
        )
        response_json["execution_result"] = cmd_result

        # Summarize the command output into a useful response
        if cmd_result.get("stdout") or cmd_result.get("stderr"):
            try:
                if source == "siri":
                    summarizer_prompt = (
                        "You ran a command for Niranjan. Analyze the output intelligently:\n"
                        "- What's the key finding?\n"
                        "- Is anything wrong or concerning?\n"
                        "- What should he know?\n"
                        "Respond in 1-3 spoken sentences. No formatting.\n"
                        "Respond with JSON: {\"speak\": \"your response\"}"
                    )
                else:
                    summarizer_prompt = (
                        "You are Cortex. Analyze this command output like a senior engineer would:\n\n"
                        "1. What does the data MEAN? Don't just list it — interpret it.\n"
                        "2. Is anything WRONG or concerning? Flag it prominently.\n"
                        "3. Are there PATTERNS or ANOMALIES?\n"
                        "4. What ACTIONABLE insight can you give?\n"
                        "5. If something needs attention, say so directly.\n\n"
                        "Use markdown. Be thorough but not verbose. Include the actual data that matters.\n"
                        "Respond with JSON: {\"speak\": \"your analysis\"}"
                    )

                summary_messages = [
                    {"role": "system", "content": summarizer_prompt},
                    {"role": "user", "content": (
                        f"User asked: {text}\n"
                        f"Command ran: {response_json['command']}\n"
                        f"Exit code: {cmd_result['exit_code']}\n"
                        f"Output:\n{cmd_result['stdout'][:2000]}\n"
                        f"Errors:\n{cmd_result['stderr'][:500]}"
                    )}
                ]
                summary_raw = llm_router.complete(summary_messages, json_mode=True, schema=SPEAK_SCHEMA)
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

    # Web search — LLM said it needs to look something up
    if response_json.get("action") == "search" and response_json.get("search_query"):
        query = response_json["search_query"]
        logger.info(f"Web search triggered: {query}")
        search_results = search_and_format(query)

        # Feed results back to LLM for a synthesized answer
        try:
            search_messages = [
                {"role": "system", "content": (
                    "You searched the web for the user. Synthesize the results into a clear, "
                    "conversational answer. Be concise. Include the key facts.\n"
                    "Respond with JSON: {\"speak\": \"your answer based on search results\"}"
                )},
                {"role": "user", "content": (
                    f"User asked: {text}\n\n"
                    f"Search results:\n{search_results}"
                )}
            ]
            search_raw = llm_router.complete(search_messages, json_mode=True, schema=SPEAK_SCHEMA)
            search_answer = json.loads(search_raw)
            if search_answer.get("speak"):
                response_json["speak"] = search_answer["speak"]
                response_json["action"] = "answer"
        except Exception as e:
            logger.error(f"Search synthesis failed: {e}")
            response_json["speak"] = "I searched but couldn't make sense of the results."

    # Create reminder
    # Reminder management — create, list, delete, edit
    if response_json.get("action") == "remind" and response_json.get("reminder"):
        reminder = response_json["reminder"]
        op = reminder.get("op", "create")

        try:
            if op == "create":
                due = datetime.fromisoformat(reminder["due"])
                reminder_id = db.create_reminder(
                    text=reminder["text"],
                    due_at=due,
                    repeat_rule=reminder.get("repeat")
                )
                response_json["reminder_id"] = reminder_id
                response_json["speak"] = response_json.get("speak", f"Reminder set: {reminder['text']}")

            elif op == "list":
                active = db.get_all_reminders(include_fired=False)
                if not active:
                    response_json["speak"] = "You have no active reminders."
                else:
                    lines = []
                    for r in active:
                        due_str = r["due_at"].strftime("%b %d, %H:%M") if hasattr(r["due_at"], "strftime") else str(r["due_at"])
                        repeat = f" (repeats {r['repeat_rule']})" if r.get("repeat_rule") else ""
                        lines.append(f"• **#{r['id']}** — {r['text']} → {due_str}{repeat}")
                    response_json["speak"] = f"You have {len(active)} reminder{'s' if len(active) != 1 else ''}:\n\n" + "\n".join(lines)

            elif op == "delete":
                rid = reminder.get("id")
                if rid:
                    success = db.delete_reminder(int(rid))
                    response_json["speak"] = f"Deleted reminder #{rid}." if success else f"No reminder #{rid} found."
                else:
                    response_json["speak"] = "Which reminder should I delete? Give me the number."

            elif op == "edit":
                rid = reminder.get("id")
                if rid:
                    due_at = datetime.fromisoformat(reminder["due"]) if reminder.get("due") else None
                    success = db.update_reminder(
                        int(rid),
                        text=reminder.get("text"),
                        due_at=due_at,
                        repeat_rule=reminder.get("repeat")
                    )
                    response_json["speak"] = f"Updated reminder #{rid}." if success else f"No reminder #{rid} found."
                else:
                    response_json["speak"] = "Which reminder should I edit? Give me the number."

        except Exception as e:
            logger.error(f"Reminder operation failed: {e}")
            response_json["speak"] = f"Reminder operation failed: {e}"

    # Browse — open URL in browser or fetch page content
    if response_json.get("action") == "browse" and response_json.get("browse_url"):
        browse_url = response_json["browse_url"]
        should_open = response_json.get("browse_open", False)

        if should_open:
            # Open in default browser
            open_result = open_in_browser(browse_url)
            response_json["speak"] = open_result["message"]
        else:
            # Fetch and read the page content
            logger.info(f"Browsing: {browse_url}")
            page_content = browse_and_format(browse_url)

            # Feed page content to LLM for synthesis
            try:
                if source == "siri":
                    browse_prompt = (
                        "You fetched a web page for Niranjan. Summarize what you found in 1-2 sentences. "
                        "Talk naturally. No formatting.\n"
                        "Respond with JSON: {\"speak\": \"your response\"}"
                    )
                else:
                    browse_prompt = (
                        "You are Cortex. You just fetched a web page for the user.\n\n"
                        "Present the content clearly. Rules:\n"
                        "- Summarize what's on the page, highlighting what the user likely cares about.\n"
                        "- If there's structured data (lists, tables, prices) — include it.\n"
                        "- Use markdown for formatting.\n"
                        "- If it's documentation — extract the relevant parts.\n"
                        "- Include key links if relevant.\n\n"
                        "Respond with JSON: {\"speak\": \"your markdown-formatted response\"}"
                    )

                browse_messages = [
                    {"role": "system", "content": browse_prompt},
                    {"role": "user", "content": (
                        f"User asked: {text}\n"
                        f"Page fetched: {browse_url}\n\n"
                        f"Page content:\n{page_content}"
                    )}
                ]
                browse_raw = llm_router.complete(browse_messages, json_mode=True, schema=SPEAK_SCHEMA)
                browse_answer = json.loads(browse_raw)
                if browse_answer.get("speak"):
                    response_json["speak"] = browse_answer["speak"]
            except Exception as e:
                logger.error(f"Browse synthesis failed: {e}")
                # Fall back to raw content summary
                response_json["speak"] = f"Here's what I found on that page:\n\n{page_content[:2000]}"

    # Crawl — authenticated multi-page crawling
    if response_json.get("action") == "crawl" and response_json.get("crawl_url"):
        crawl_url = response_json["crawl_url"]
        cred_name = response_json.get("crawl_credential")
        max_pages = response_json.get("crawl_max_pages", 3)

        # Resolve credentials
        credentials = {}
        if cred_name:
            credentials = resolve_credential(cred_name)
            if not credentials:
                response_json["speak"] = (
                    f"I don't have credentials named '{cred_name}' configured. "
                    f"Add them to config.yaml under web_credentials."
                )
                # Skip crawl, just return the error
                memory_mgr.on_conversation_turn(session_id)
                return jsonify(response_json)

        logger.info(f"Crawling: {crawl_url} (cred={cred_name}, max_pages={max_pages})")
        crawl_content = crawl_and_format(crawl_url, credentials, max_pages=max_pages)

        # Feed crawl results to LLM for synthesis
        try:
            if source == "siri":
                crawl_prompt = (
                    "You crawled a website for Niranjan. Summarize what you found in 2-3 sentences. "
                    "Talk naturally. No formatting.\n"
                    "Respond with JSON: {\"speak\": \"your response\"}"
                )
            else:
                crawl_prompt = (
                    "You are Cortex. You just crawled a website for the user (possibly behind auth).\n\n"
                    "Present the findings clearly. Rules:\n"
                    "- Summarize what's across the pages you crawled.\n"
                    "- If there's structured data — present it.\n"
                    "- Highlight anything notable, errors, or actionable items.\n"
                    "- Use markdown.\n\n"
                    "Respond with JSON: {\"speak\": \"your markdown-formatted response\"}"
                )

            crawl_messages = [
                {"role": "system", "content": crawl_prompt},
                {"role": "user", "content": (
                    f"User asked: {text}\n"
                    f"Crawled: {crawl_url} ({max_pages} pages max)\n"
                    f"Credential used: {cred_name or 'none (public)'}\n\n"
                    f"Crawl results:\n{crawl_content}"
                )}
            ]
            crawl_raw = llm_router.complete(crawl_messages, json_mode=True, schema=SPEAK_SCHEMA)
            crawl_answer = json.loads(crawl_raw)
            if crawl_answer.get("speak"):
                response_json["speak"] = crawl_answer["speak"]
        except Exception as e:
            logger.error(f"Crawl synthesis failed: {e}")
            response_json["speak"] = f"Crawled {crawl_url}. Here's what I got:\n\n{crawl_content[:2000]}"

    # Credential prompt — trigger macOS native dialog for username/password
    if response_json.get("action") == "credential_prompt" and response_json.get("credential_prompt_site"):
        site_name = response_json["credential_prompt_site"]
        prompt_for = response_json.get("credential_prompt_for", "")

        logger.info(f"Credential prompt triggered for: {site_name}")
        response_json["speak"] = f"I need credentials for {site_name}. A login dialog should pop up on your screen."

        # Show native macOS dialog
        cred_result = prompt_credentials(site_name=site_name)

        if cred_result["cancelled"]:
            response_json["speak"] = f"No worries, cancelled the login for {site_name}."
        elif cred_result["success"]:
            # Got credentials — now retry the original intent
            # Build an ad-hoc credential dict for the crawl
            runtime_cred = {
                "auth_type": "form",
                "username": cred_result["username"],
                "password": cred_result["password"],
            }

            # If the original intent was to crawl, do it now
            if "crawl" in prompt_for.lower() or "http" in prompt_for.lower():
                # Extract URL from the prompt_for field
                import re as _re
                url_match = _re.search(r'https?://\S+', prompt_for)
                if url_match:
                    crawl_url = url_match.group(0)
                    logger.info(f"Retrying crawl with prompted credentials: {crawl_url}")
                    crawl_content = crawl_and_format(crawl_url, runtime_cred, max_pages=3)
                    try:
                        crawl_messages = [
                            {"role": "system", "content": (
                                "You are Cortex. You just crawled a site after getting login credentials.\n"
                                "Present the findings clearly in markdown.\n"
                                "Respond with JSON: {\"speak\": \"your response\"}"
                            )},
                            {"role": "user", "content": (
                                f"User originally asked: {text}\n"
                                f"Crawled: {crawl_url}\n\n"
                                f"Results:\n{crawl_content}"
                            )}
                        ]
                        raw = llm_router.complete(crawl_messages, json_mode=True, schema=SPEAK_SCHEMA)
                        answer = json.loads(raw)
                        if answer.get("speak"):
                            response_json["speak"] = answer["speak"]
                    except Exception as e:
                        logger.error(f"Post-credential crawl synthesis failed: {e}")
                        response_json["speak"] = f"Logged in. Here's what I found:\n\n{crawl_content[:2000]}"
                else:
                    response_json["speak"] = f"Got your credentials for {site_name}. But I couldn't determine which URL to crawl."
            else:
                response_json["speak"] = f"Got your credentials for {site_name}. Saved for this session."
        else:
            response_json["speak"] = f"Something went wrong with the login dialog for {site_name}."

    # Edit file — read file, generate edit via LLM, apply it
    if response_json.get("action") == "edit" and response_json.get("edit_file"):
        edit_filepath = response_json["edit_file"]
        edit_instruction = response_json.get("edit_instruction", "")

        logger.info(f"File edit triggered: {edit_filepath} — {edit_instruction[:80]}")

        # If the path is just a filename (no directory), try to find it
        if "/" not in edit_filepath and "\\" not in edit_filepath:
            found = find_file(edit_filepath)
            if found:
                edit_filepath = found[0]  # Use first match
                logger.info(f"Resolved file path: {edit_filepath}")
            else:
                response_json["speak"] = f"I couldn't find a file named '{edit_filepath}'. Can you give me the full path?"
                memory_mgr.on_conversation_turn(session_id)
                return jsonify(response_json)

        # Read the file
        file_data = editor_read_file(edit_filepath)
        if not file_data["success"]:
            response_json["speak"] = f"Can't read {edit_filepath}: {file_data['error']}"
            memory_mgr.on_conversation_turn(session_id)
            return jsonify(response_json)

        # Ask LLM to generate the edit
        try:
            edit_messages = [
                {"role": "system", "content": (
                    "You are a code editor. Given a file and an edit instruction, produce the exact edit.\n\n"
                    "Respond with JSON ONLY:\n"
                    "{\n"
                    '  "find": "exact text to find in the file (copy verbatim, including whitespace)",\n'
                    '  "replace": "the replacement text",\n'
                    '  "explanation": "one-line explanation of what you changed"\n'
                    "}\n\n"
                    "RULES:\n"
                    "- The 'find' field must be an EXACT substring of the file content.\n"
                    "- Include enough context in 'find' to be unique (a few lines before/after the change).\n"
                    "- Keep changes minimal — only change what's needed.\n"
                    "- Preserve indentation and style.\n"
                    "- If multiple edits needed, return a JSON array of edit objects."
                )},
                {"role": "user", "content": (
                    f"FILE: {edit_filepath}\n"
                    f"LANGUAGE: {file_data.get('language', 'unknown')}\n"
                    f"INSTRUCTION: {edit_instruction}\n\n"
                    f"FILE CONTENT:\n```\n{file_data['content'][:15000]}\n```"
                )}
            ]
            # No schema= here deliberately: this prompt allows either a single edit
            # object or an array of them, and this path already sits behind the
            # destructive-action confirmation gate, so a malformed response fails
            # safely (caught below) rather than silently doing the wrong thing.
            edit_raw = llm_router.complete(edit_messages, json_mode=True)
            edit_data = json.loads(edit_raw)

            # Handle single edit or array of edits
            if isinstance(edit_data, list):
                edits = [{"find": e["find"], "replace": e["replace"]} for e in edit_data]
                result = apply_patch(edit_filepath, edits)
                explanation = "; ".join(e.get("explanation", "") for e in edit_data)
            else:
                result = apply_edit(edit_filepath, edit_data["find"], edit_data["replace"])
                explanation = edit_data.get("explanation", "")

            if result["success"]:
                diff_preview = result.get("diff", "")[:1000]
                if source == "siri":
                    response_json["speak"] = f"Done. {explanation}"
                else:
                    response_json["speak"] = f"**Edited** `{edit_filepath}`\n\n{explanation}\n\n```diff\n{diff_preview}\n```"
            else:
                response_json["speak"] = f"Edit failed: {result.get('error', 'unknown error')}. The file wasn't changed."

        except Exception as e:
            logger.error(f"File edit failed: {e}")
            response_json["speak"] = f"I tried to edit the file but something went wrong: {e}"

    # Run project — detect project type and execute the right command
    if response_json.get("action") == "run_project" and response_json.get("run_project_path"):
        project_path = response_json["run_project_path"]
        run_command = response_json.get("run_project_command", "start")

        logger.info(f"Run project: {project_path} — command: {run_command}")

        # Detect project type
        project_info = detect_project(project_path)

        if project_info.get("error") and "not found" in project_info["error"].lower():
            response_json["speak"] = f"Can't find a project at `{project_path}`. Check the path?"
            memory_mgr.on_conversation_turn(session_id)
            return jsonify(response_json)

        # Resolve the actual shell command
        if run_command in project_info.get("runners", {}):
            actual_command = project_info["runners"][run_command]
        else:
            # Treat it as a custom command
            actual_command = run_command

        logger.info(f"Running: {actual_command} in {project_path}")

        # Execute it
        cmd_result = executor.execute(
            command=actual_command,
            cwd=project_path,
            timeout=30,
            source=source
        )
        response_json["execution_result"] = cmd_result

        # Summarize
        if source == "siri":
            if cmd_result["exit_code"] == 0:
                response_json["speak"] = f"Started. The {project_info.get('type', 'project')} is running."
            else:
                response_json["speak"] = f"Something went wrong running the project. Exit code {cmd_result['exit_code']}."
        else:
            if cmd_result["exit_code"] == 0:
                output_preview = cmd_result.get("stdout", "")[:500]
                response_json["speak"] = (
                    f"**Running** `{actual_command}` in `{project_path}`\n\n"
                    f"Project type: {project_info.get('type', 'unknown')}\n\n"
                    f"```\n{output_preview}\n```"
                )
            else:
                stderr = cmd_result.get("stderr", "")[:500]
                response_json["speak"] = (
                    f"**Failed** — `{actual_command}` exited with code {cmd_result['exit_code']}\n\n"
                    f"```\n{stderr}\n```"
                )

    # ─── Log experience (State-Action-Reward buffer) — off the response path ───
    _bg.submit(_log_experience_bg, session_id, text, response_json)

    # ─── Memory extraction (genuinely async — see memory.py) ───
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
        parsed = llm_router.complete(parse_messages, json_mode=True, schema=REMINDER_PARSE_SCHEMA)
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


@app.route("/reminders/<int:reminder_id>", methods=["DELETE"])
def delete_reminder(reminder_id):
    """Delete a reminder by ID."""
    success = db.delete_reminder(reminder_id)
    if success:
        return jsonify({"speak": f"Deleted reminder #{reminder_id}.", "success": True})
    return jsonify({"speak": f"No reminder found with ID {reminder_id}.", "success": False}), 404


@app.route("/reminders/<int:reminder_id>", methods=["PUT"])
def update_reminder(reminder_id):
    """Update a reminder (text, due_at, repeat_rule)."""
    data = request.get_json(force=True)
    due_at = None
    if data.get("due_at"):
        due_at = datetime.fromisoformat(data["due_at"])
    success = db.update_reminder(
        reminder_id,
        text=data.get("text"),
        due_at=due_at,
        repeat_rule=data.get("repeat_rule")
    )
    if success:
        return jsonify({"speak": f"Updated reminder #{reminder_id}.", "success": True})
    return jsonify({"speak": f"No reminder found with ID {reminder_id}.", "success": False}), 404


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
# AUTONOMY
# ═══════════════════════════════════════════════════════════════

@app.route("/autonomy/log", methods=["GET"])
def get_autonomy_log():
    """Recent autonomy loop cycles, newest first."""
    limit = int(request.args.get("limit", 20))
    return jsonify({"log": db.get_autonomy_log(limit=limit)})


@app.route("/autonomy/goals", methods=["GET"])
def get_goals():
    """Active goals — self-formed or user-seeded."""
    return jsonify({"goals": db.get_active_goals(limit=int(request.args.get("limit", 20)))})


@app.route("/autonomy/goals", methods=["POST"])
def add_goal():
    """Seed a goal for the autonomy loop to work toward."""
    data = request.get_json(force=True)
    description = data.get("description", "").strip()
    if not description:
        return jsonify({"error": "description required"}), 400
    goal_id = db.create_goal(description=description, priority=int(data.get("priority", 3)), source="user")
    return jsonify({"speak": "Got it, I'll work toward that.", "goal_id": goal_id})


@app.route("/autonomy/priorities", methods=["GET"])
def get_priorities():
    """Standing priorities — the compass shared by /ask and the autonomy loop."""
    return jsonify({"priorities": db.get_active_priorities(limit=int(request.args.get("limit", 20)))})


@app.route("/autonomy/priorities", methods=["POST"])
def add_priority():
    """Seed a standing priority (weight 1-5)."""
    data = request.get_json(force=True)
    statement = data.get("statement", "").strip()
    if not statement:
        return jsonify({"error": "statement required"}), 400
    priority_id = db.create_priority(statement=statement, weight=int(data.get("weight", 3)), source="user")
    return jsonify({"speak": "Got it, I'll weigh that going forward.", "priority_id": priority_id})


@app.route("/autonomy/priorities/<int:priority_id>/retire", methods=["POST"])
def retire_priority(priority_id):
    """Retire a standing priority (soft delete — keeps history)."""
    success = db.retire_priority(priority_id)
    return jsonify({"speak": "Retired." if success else f"No priority #{priority_id} found."})


@app.route("/autonomy/proposals/<int:proposal_id>/confirm", methods=["POST"])
def confirm_autonomy_proposal(proposal_id):
    """Approve an action the autonomy loop queued for confirmation."""
    proposal = db.get_pending_proposal(proposal_id)
    if not proposal:
        return jsonify({"error": "No pending proposal with that id"}), 404

    held = proposal["action_payload"] or {}
    speak, cmd_result = _execute_held_action(held, source="autonomy")
    exit_code = cmd_result.get("exit_code") if cmd_result else None
    db.resolve_proposal(proposal_id, confirmed=True, exit_code=exit_code)
    return jsonify({"speak": speak})


@app.route("/autonomy/proposals/<int:proposal_id>/deny", methods=["POST"])
def deny_autonomy_proposal(proposal_id):
    """Reject an action the autonomy loop queued for confirmation."""
    proposal = db.get_pending_proposal(proposal_id)
    if not proposal:
        return jsonify({"error": "No pending proposal with that id"}), 404
    db.resolve_proposal(proposal_id, confirmed=False)
    return jsonify({"speak": "Cancelled."})


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
        "llm_primary": f"ollama/{config['llm'].get('ollama', {}).get('model', 'gemma4:e4b')} (local)",
        "llm_fallback": f"ollama/{config['llm'].get('ollama_fallback', {}).get('model')} (local)" if config['llm'].get('ollama_fallback') else None,
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
    import os as _os
    import sys as _sys
    logger.info("Reload requested — restarting process...")
    # Restart by re-execing the same process
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)


# ═══════════════════════════════════════════════════════════════
# VISION (Screen Capture + Analysis)
# ═══════════════════════════════════════════════════════════════

@app.route("/vision", methods=["POST"])
def vision():
    """Capture screen and analyze with GPT-4o vision."""
    data = request.get_json(force=True) if request.data else {}
    question = data.get("question", "What's on this screen? What should I know?")
    source = data.get("source", "api")

    if source == "siri":
        result = analyze_screen_for_voice(question=question)
    else:
        result = analyze_screen(question=question)

    return jsonify({"speak": result, "action": "answer"})


# ═══════════════════════════════════════════════════════════════
# TODOS (AppFlowy)
# ═══════════════════════════════════════════════════════════════

@app.route("/todos", methods=["GET"])
def get_todos():
    """List todos from AppFlowy."""
    if not appflowy_client:
        return jsonify({"error": "AppFlowy not configured", "speak": "AppFlowy isn't set up yet."}), 400

    todos = appflowy_client.get_todos()
    formatted = appflowy_client.format_todos(todos)
    return jsonify({"todos": todos, "formatted": formatted, "count": len(todos)})


@app.route("/todos", methods=["POST"])
def add_todo():
    """Add a todo to AppFlowy."""
    if not appflowy_client:
        return jsonify({"error": "AppFlowy not configured"}), 400

    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400

    result = appflowy_client.add_todo(title, extra_fields=data.get("fields"))
    return jsonify({"speak": f"Added: {title}", "result": result})


@app.route("/todos/complete", methods=["POST"])
def complete_todo():
    """Mark a todo as complete."""
    if not appflowy_client:
        return jsonify({"error": "AppFlowy not configured"}), 400

    data = request.get_json(force=True)
    row_id = data.get("row_id", "").strip()
    if not row_id:
        return jsonify({"error": "No row_id provided"}), 400

    result = appflowy_client.complete_todo(row_id)
    return jsonify({"speak": "Marked as done.", "result": result})


@app.route("/todos/setup", methods=["GET"])
def setup_todos():
    """Discovery endpoint — list workspaces and databases to help configure."""
    if not appflowy_client:
        return jsonify({"error": "AppFlowy not configured. Add email/password to config.yaml"}), 400

    workspaces = appflowy_client.get_workspaces()
    databases = appflowy_client.get_databases() if appflowy_client._workspace_id else []
    folder = appflowy_client.get_folder() if appflowy_client._workspace_id else {}

    return jsonify({
        "workspaces": workspaces,
        "databases": databases,
        "folder": folder,
        "instructions": "Set workspace_id and todo_database_id in config.yaml"
    })


# ═══════════════════════════════════════════════════════════════
# REPORTS (Conversational delivery)
# ═══════════════════════════════════════════════════════════════

@app.route("/reports/deliver", methods=["POST"])
def deliver_report():
    """User said yes — deliver the pending report."""
    data = request.get_json(force=True) if request.data else {}
    name = data.get("name")
    content = report_scheduler.deliver_report(report_name=name)

    # Speak it
    from nlg import to_speech
    subprocess.Popen(
        ["say", "-v", "Tara", "-r", "175", to_speech(content)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return jsonify({"speak": content})


@app.route("/reports/later", methods=["POST"])
def defer_report():
    """User said later — reschedule."""
    data = request.get_json(force=True) if request.data else {}
    minutes = data.get("minutes", 30)
    name = data.get("name")
    result = report_scheduler.reschedule(minutes=minutes, report_name=name)
    return jsonify({"speak": result})


@app.route("/reports/pending", methods=["GET"])
def pending_reports():
    """Check if reports are waiting."""
    names = report_scheduler.get_pending_names()
    return jsonify({"pending": names, "count": len(names)})


# ═══════════════════════════════════════════════════════════════
# CALENDAR
# ═══════════════════════════════════════════════════════════════

@app.route("/calendar", methods=["GET"])
def calendar():
    """Get calendar events."""
    scope = request.args.get("scope", "today")  # today | upcoming | tomorrow
    if scope == "upcoming":
        hours = int(request.args.get("hours", 4))
        events = get_upcoming_events(hours)
    elif scope == "tomorrow":
        from calendar_integration import get_tomorrows_events
        events = get_tomorrows_events()
    else:
        events = get_todays_events()

    formatted = format_events_for_context(events)
    return jsonify({"events": events, "formatted": formatted, "count": len(events)})


# ═══════════════════════════════════════════════════════════════
# REMOTE / MULTI-MACHINE
# ═══════════════════════════════════════════════════════════════

@app.route("/remote/execute", methods=["POST"])
def remote_execute():
    """Execute a command on a remote host."""
    data = request.get_json(force=True)
    host = data.get("host", "").strip()
    command = data.get("command", "").strip()
    if not host or not command:
        return jsonify({"error": "host and command required"}), 400

    result = remote_executor.execute(host, command, timeout=data.get("timeout", 30))
    return jsonify(result)


@app.route("/remote/hosts", methods=["GET"])
def remote_hosts():
    """List configured remote hosts + tailscale peers."""
    configured = remote_executor.list_hosts()
    peers = get_tailscale_peers()
    return jsonify({"configured": configured, "tailscale_peers": peers})


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Explicit feedback on the most recent turn for a session.
    Overrides heuristic reward signals — this is the strongest signal in the
    reward function (see db.compute_pending_rewards).
    Body: {"session_id": "...", "rating": -1 | 0 | 1}
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    rating = data.get("rating")

    if not session_id or rating not in (-1, 0, 1):
        return jsonify({"error": "session_id and rating (-1|0|1) required"}), 400

    exp_id = _last_experience.get(session_id)
    if not exp_id:
        exp_id = db.find_recent_experience(session_id, "")
    if not exp_id:
        return jsonify({"speak": "No recent interaction to rate.", "success": False}), 404

    db.update_experience_outcome(exp_id, explicit_feedback=rating)
    db.compute_pending_rewards()
    return jsonify({"speak": "Thanks, noted.", "success": True, "experience_id": exp_id})


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    """
    Generate TTS audio via macOS's local `say` command. Fully local, zero network,
    zero API key. Returns raw PCM 24kHz 16-bit mono (converted from say's AIFF output).
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    try:
        import tempfile
        import wave
        import audioop

        from nlg import to_speech
        clean_text = to_speech(text)

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_aiff:
            aiff_path = tmp_aiff.name

        subprocess.run(
            ["say", "-v", "Samantha", "-o", aiff_path, clean_text],
            check=True, capture_output=True, timeout=30
        )

        # Read the AIFF and resample to 24kHz 16-bit mono PCM to match what
        # voice.py's playback path expects (same contract as the old endpoint).
        import aifc
        with aifc.open(aiff_path, "rb") as af:
            n_channels = af.getnchannels()
            frame_rate = af.getframerate()
            n_frames = af.getnframes()
            raw = af.readframes(n_frames)

        os.unlink(aiff_path)

        # Downmix to mono if needed
        if n_channels == 2:
            raw = audioop.tomono(raw, 2, 0.5, 0.5)

        # Resample to 24000 Hz to match the contract voice.py expects
        target_rate = 24000
        if frame_rate != target_rate:
            raw, _ = audioop.ratecv(raw, 2, 1, frame_rate, target_rate, None)

        from flask import Response
        return Response(raw, mimetype="audio/pcm")
    except Exception as e:
        logger.error(f"Local TTS failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/stt", methods=["POST"])
def stt_endpoint():
    """Transcribe audio via Google Speech Recognition. Accepts raw PCM 16-bit mono."""
    try:
        import tempfile
        import wave
        import numpy as np
        import speech_recognition as sr

        pcm_data = request.get_data()
        sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
        duration = len(pcm_data) / (2 * sample_rate)

        if not pcm_data or len(pcm_data) < 3200:
            return jsonify({"text": ""})

        # Analyze + normalize amplitude (mic input is often quiet)
        arr = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        peak = np.abs(arr).max()
        rms = np.sqrt(np.mean(arr ** 2))
        logger.info(f"STT: {len(pcm_data)} bytes, {sample_rate}Hz, {duration:.2f}s, peak={peak:.0f}, rms={rms:.0f}")

        # Normalize to ~80% of int16 range so Google hears it clearly
        if peak > 0:
            arr = arr / peak * (32767 * 0.8)
        arr = arr.astype(np.int16)
        normalized_pcm = arr.tobytes()

        # Save for inspection
        debug_path = os.path.join(os.path.dirname(__file__), "logs", "last_stt.wav")
        with wave.open(debug_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(normalized_pcm)

        # Transcribe with Google
        recognizer = sr.Recognizer()
        with sr.AudioFile(debug_path) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio)
            logger.info(f"STT result: '{text}'")
            return jsonify({"text": text})
        except sr.UnknownValueError:
            logger.info(f"STT: could not understand (peak={peak:.0f}, rms={rms:.0f})")
            return jsonify({"text": ""})
        except sr.RequestError as e:
            logger.error(f"Google STT request error: {e}")
            return jsonify({"text": "", "error": str(e)})

    except Exception as e:
        logger.error(f"STT failed: {e}")
        return jsonify({"text": "", "error": str(e)}), 500


# ─── Entry Point ───
if __name__ == "__main__":
    port = config["server"]["port"]
    host = config["server"]["host"]
    logger.info(f"Cortex daemon starting on {host}:{port}")
    logger.info(f"Memory: max_facts={memory_mgr.max_facts}, extract_every={memory_mgr.summarize_every} turns")
    logger.info(f"Reminders: checking every {reminder_mgr.check_interval}s")
    app.run(host=host, port=port, debug=False, threaded=True)
