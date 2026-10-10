"""
JSON Schemas for constrained LLM decoding — narrow structured-extraction
sub-tasks (parse a reminder, extract facts, generate a devops plan, produce
an edit, summarize into one "speak" field). Passed as the `format` field to
Ollama's /api/chat (see llm.py), which does grammar-constrained generation:
the model can ONLY emit JSON matching the exact shape, not just "some
syntactically valid JSON" (all json_mode=True alone guarantees).

The top-level "what should I do" action decision does NOT use these — that
uses native tool-calling instead (tools.py), which fits an open-ended
decision much better than asking the model to author one giant JSON blob.
Reserve this file for tasks that really are "extract data into this shape."
"""

# The many single-field "Respond with JSON: {\"speak\": ...}" summarizer prompts
# scattered through cortex.py (news synthesis, browse/crawl/output summaries).
SPEAK_SCHEMA = {
    "type": "object",
    "properties": {"speak": {"type": "string"}},
    "required": ["speak"],
}

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "find": {"type": "string"},
        "replace": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["find", "replace", "explanation"],
}

REMINDER_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "due": {"type": "string"},
        "repeat": {"type": ["string", "null"]},
    },
    "required": ["text", "due"],
}

FACTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "type": {"type": "string", "enum": ["fact", "preference", "learned_command", "project_info"]},
            "importance": {"type": "integer"},
        },
        "required": ["content", "type", "importance"],
    },
}

DEVOPS_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "speak": {"type": "string"},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "cmd": {"type": "string"},
                    "desc": {"type": "string"},
                    "critical": {"type": "boolean"},
                },
                "required": ["step", "cmd", "desc"],
            },
        },
        "rollback": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["speak", "plan"],
}

DEVOPS_RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["retry", "skip", "abort"]},
        "new_cmd": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

AUTONOMY_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "think": {"type": "string"},
        "decision": {"type": "string", "enum": ["noop", "new_goal", "act"]},
        "goal": {"type": ["string", "null"]},
        "action": {
            "type": ["object", "null"],
            "properties": {
                "action": {"type": "string", "enum": ["execute", "answer", "remind"]},
                "command": {"type": ["string", "null"]},
                "reminder": {
                    "type": ["object", "null"],
                    "properties": {
                        "op": {"type": "string"},
                        "text": {"type": ["string", "null"]},
                        "due": {"type": ["string", "null"]},
                        "repeat": {"type": ["string", "null"]},
                    },
                },
                "speak": {"type": "string"},
            },
        },
    },
    "required": ["think", "decision"],
}

PRIORITY_FORMATION_SCHEMA = {
    "type": "object",
    "properties": {
        "promote": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "weight": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["statement", "weight", "reason"],
            },
        },
        "skipped_reason": {"type": "string"},
    },
    "required": ["promote"],
}
