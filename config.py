"""Configuration and tool schemas."""
import os
from pathlib import Path

from todos import (
    MAX_TODO_EXPLANATION_CHARS,
    MAX_TODO_ITEMS,
    MAX_TODO_STEP_CHARS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHAT_PROVIDERS = tuple(
    name.strip()
    for name in os.getenv(
        "CHAT_PROVIDERS",
        "telegram,responses_api",
    ).split(",")
    if name.strip()
)
if not CHAT_PROVIDERS:
    raise ValueError("CHAT_PROVIDERS must contain at least one provider")

# Inbound, OpenAI Responses-compatible chat provider.  This is deliberately
# separate from RESPONSES_BASE_URL, which is the model backend the agent calls.
RESPONSES_SERVER_HOST = os.getenv(
    "RESPONSES_SERVER_HOST",
    "0.0.0.0",
).strip()
RESPONSES_SERVER_PORT = int(os.getenv("RESPONSES_SERVER_PORT", "8787"))
RESPONSES_SERVER_API_KEY = os.getenv(
    "RESPONSES_SERVER_API_KEY",
    "strong-secret",
)
RESPONSES_SERVER_MAX_BODY = max(
    1024,
    int(os.getenv("RESPONSES_SERVER_MAX_BODY", "1048576")),
)

# Telegram adapter configuration. These are optional at framework import
# time and validated only when the Telegram provider is initialized.
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_USER_ID = int(os.getenv("TG_USER_ID", "0"))

RESPONSES_BASE_URL = os.getenv(
    "RESPONSES_BASE_URL",
    "http://127.0.0.1:8000/v1",
).rstrip("/")

RESPONSES_API_KEY = os.getenv("RESPONSES_API_KEY", "")
RESPONSES_MODEL = os.getenv("RESPONSES_MODEL", "")
RESPONSES_REASONING_EFFORT = (
    os.getenv("RESPONSES_REASONING_EFFORT", "").strip() or None
)
RESPONSES_REASONING_SUMMARY = (
    os.getenv("RESPONSES_REASONING_SUMMARY", "auto").strip() or None
)
RESPONSES_MAX_RETRIES = max(
    0,
    int(os.getenv("RESPONSES_MAX_RETRIES", "3")),
)
RESPONSES_RETRY_BASE_SECONDS = max(
    0.0,
    float(os.getenv("RESPONSES_RETRY_BASE_SECONDS", "0.5")),
)

SUBAGENT_RESPONSES_BASE_URL = os.getenv(
    "SUBAGENT_RESPONSES_BASE_URL",
    "http://127.0.0.1:8000/v1",
).rstrip("/")

SUBAGENT_RESPONSES_API_KEY = os.getenv(
    "SUBAGENT_RESPONSES_API_KEY",
    "",
)

SUBAGENT_RESPONSES_MODEL = os.getenv(
    "SUBAGENT_RESPONSES_MODEL",
    "",
)

SUBAGENT_RESPONSES_REASONING_EFFORT = (
    os.getenv(
        "SUBAGENT_RESPONSES_REASONING_EFFORT",
        "",
    ).strip()
    or None
)
SUBAGENT_RESPONSES_MAX_RETRIES = max(
    0,
    int(os.getenv("SUBAGENT_RESPONSES_MAX_RETRIES", "3")),
)
SUBAGENT_RESPONSES_RETRY_BASE_SECONDS = max(
    0.0,
    float(os.getenv("SUBAGENT_RESPONSES_RETRY_BASE_SECONDS", "0.5")),
)

MAX_SUBAGENT_DEPTH = max(
    1,
    int(
        os.getenv(
            "MAX_SUBAGENT_DEPTH",
            "3",
        )
    ),
)

MAX_SUBAGENT_RECORDS = max(
    1,
    int(os.getenv("MAX_SUBAGENT_RECORDS", "100")),
)
MAX_SUBAGENT_TOOL_EVENTS = max(
    1,
    int(os.getenv("MAX_SUBAGENT_TOOL_EVENTS", "500")),
)
MAX_SUBAGENT_PENDING_INPUTS = max(
    1,
    int(os.getenv("MAX_SUBAGENT_PENDING_INPUTS", "64")),
)
MAX_BACKGROUND_SUBAGENTS = max(
    1,
    int(os.getenv("MAX_BACKGROUND_SUBAGENTS", "16")),
)

AGENT_WORKDIR = Path(
    os.getenv(
        "AGENT_WORKDIR",
        ".",
    )
).expanduser().resolve()

RAPTOR_HOME = Path(
    os.getenv(
        "RAPTOR_HOME",
        str(AGENT_WORKDIR / ".raptor"),
    )
).expanduser().resolve()

CHAT_DIR = RAPTOR_HOME / "chats"

STATE_PATH = RAPTOR_HOME / "state.json"

LOG_PATH = Path(
    os.getenv(
        "RAPTOR_LOG",
        str(RAPTOR_HOME / "raptor.log"),
    )
).expanduser().resolve()

MAX_TOOL_ROUNDS = max(
    0,
    int(
        os.getenv(
            "MAX_TOOL_ROUNDS",
            "0",
        )
    ),
)
MAX_PENDING_STEERS = max(
    1,
    int(os.getenv("MAX_PENDING_STEERS", "64")),
)

SHELL_TIMEOUT = int(
    os.getenv(
        "SHELL_TIMEOUT",
        "120",
    )
)

MAX_TOOL_OUTPUT = int(
    os.getenv(
        "MAX_TOOL_OUTPUT",
        "30000",
    )
)

COMPACTION_OUTPUT_TOKENS = max(
    256,
    int(
        os.getenv(
            "COMPACTION_OUTPUT_TOKENS",
            "4000",
        )
    ),
)

# The Responses API counts hidden reasoning and visible checkpoint text against
# the same generation limit.  Keep the durable checkpoint cap independent from
# the model's generation allowance so a reasoning model cannot consume the
# entire budget before emitting its summary.
COMPACTION_GENERATION_TOKENS = max(
    COMPACTION_OUTPUT_TOKENS,
    int(
        os.getenv(
            "COMPACTION_GENERATION_TOKENS",
            str(COMPACTION_OUTPUT_TOKENS + 4096),
        )
    ),
)

COMPACTION_REASONING_EFFORT = (
    os.getenv("COMPACTION_REASONING_EFFORT", "low").strip() or None
)

MODEL_CONTEXT_TOKENS = max(
    0,
    int(
        os.getenv(
            "MODEL_CONTEXT_TOKENS",
            "0",
        )
    ),
)

SUBAGENT_MODEL_CONTEXT_TOKENS = max(
    0,
    int(
        os.getenv(
            "SUBAGENT_MODEL_CONTEXT_TOKENS",
            "0",
        )
    ),
)

COMPACT_KEEP_RECENT_TOKENS = max(
    0,
    int(
        os.getenv(
            "COMPACT_KEEP_RECENT_TOKENS",
            "20000",
        )
    ),
)

# Original user requests retained alongside a generated checkpoint.  These are
# semantic anchors, not another model-written summary.
COMPACTION_USER_ANCHOR_TOKENS = max(
    0,
    int(
        os.getenv(
            "COMPACTION_USER_ANCHOR_TOKENS",
            "20000",
        )
    ),
)

# Per-record char cap when rendering archive items into a checkpoint
# request. Canonical JSONL is never truncated.
COMPACTION_MAX_RECORD_CHARS = max(
    1024,
    int(
        os.getenv(
            "COMPACTION_MAX_RECORD_CHARS",
            "12000",
        )
    ),
)

CONTEXT_COMPACT_RATIO = min(
    0.95,
    max(
        0.50,
        float(
            os.getenv(
                "CONTEXT_COMPACT_RATIO",
                "0.82",
            )
        ),
    ),
)

CONTEXT_SAFETY_TOKENS = max(
    0,
    int(
        os.getenv(
            "CONTEXT_SAFETY_TOKENS",
            "4096",
        )
    ),
)


def _context_input_budget(model_context_tokens: int) -> int:
    if not model_context_tokens:
        return 0

    ratio_budget = int(
        model_context_tokens
        * CONTEXT_COMPACT_RATIO
    )

    safety_budget = max(
        1,
        model_context_tokens
        - CONTEXT_SAFETY_TOKENS,
    )

    return min(
        ratio_budget,
        safety_budget,
    )


def context_input_budget() -> int:
    return _context_input_budget(MODEL_CONTEXT_TOKENS)


def subagent_context_input_budget() -> int:
    return _context_input_budget(SUBAGENT_MODEL_CONTEXT_TOKENS)


def _compaction_generation_budget(model_context_tokens: int) -> int:
    """Generation allowance that remains sane on smaller context models."""
    if not model_context_tokens:
        return COMPACTION_GENERATION_TOKENS
    return min(
        COMPACTION_GENERATION_TOKENS,
        max(1024, model_context_tokens // 4),
    )


def compaction_generation_budget() -> int:
    return _compaction_generation_budget(MODEL_CONTEXT_TOKENS)


def subagent_compaction_generation_budget() -> int:
    return _compaction_generation_budget(SUBAGENT_MODEL_CONTEXT_TOKENS)


CHAT_STREAMING = (
    os.getenv("CHAT_STREAMING", "1").lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

TELEGRAM_MARKDOWN = (
    os.getenv(
        "TELEGRAM_MARKDOWN",
        "1",
    ).lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

CHAT_STREAM_INTERVAL = float(
    os.getenv("CHAT_STREAM_INTERVAL", "0.35")
)

TG_API = (
    f"https://api.telegram.org/"
    f"bot{TG_BOT_TOKEN}"
)


# ---------------------------------------------------------------------------
# Agent instructions / tools
# ---------------------------------------------------------------------------

BASE_INSTRUCTIONS = f"""You are a concise local coding assistant available
through interactive chat providers.

IDENTITY: read AGENTS.md and MEMORY.md from the workspace root. Internalize
the identity, conventions, and working style they define. Do not narrate
reading them back to the user.

TOOLS:
- persistent todos
- shell execution
- file reading
- file writing
- exact file editing
- directory listing
- foreground and background subagents
- chat_history for archived lossless transcripts
- get_goal / set_goal / update_goal for durable root goals

WORKFLOW:
- On a task: inspect relevant files first, make focused changes, run useful
  tests or checks, and report the result.
- Use the update_plan tool for genuinely multi-step work (pending / in_progress /
  completed). Skip todos for trivial one-step questions.
- Treat update_plan as a complete ordered snapshot, not item-by-item CRUD.
  Keep at most one step in_progress; mark a finished step completed as soon as
  it is verified, and move the next step to in_progress in the same update.
- Todos track execution progress. They do not prove that a persistent goal is
  complete; call update_goal only after the full goal outcome is achieved and
  verified.
- Use the subagent tool only when the user explicitly requests delegation.
  Subagents share your workspace but have isolated context. You are the only
  agent that communicates with the user.
- Use set_goal only when the current user explicitly requests
  persistent/autonomous goal execution. Never infer goal creation merely
  from a difficult, long, or multi-step task. If the user declines or does
  not request persistence, do not call it.
- Never claim a tool action succeeded unless its result confirms it.

HISTORY:
- Your active conversation context may contain a compacted checkpoint rather
  than the complete session transcript.
- Raptor keeps the complete lossless transcript in its chat archive.
- Use the chat_history tool when an exact earlier detail, command, decision,
  result, path, constraint, or user statement is needed but is absent from the
  active checkpoint.
- Do not guess missing historical details when they can be retrieved.
- Prefer the current session. Search older sessions only when relevant.
- Do not search history unnecessarily when the active context already contains
  what you need.

FIRST MESSAGE:
When the operator opens a new session (sends a greeting or "hi"), respond
with a short greeting only — one line. Do not run boot checks, do not
report container or SearXNG status, do not mention MEMORY.md or todos,
and do not ask "what are we working on?". Just greet and wait.

Filesystem tools are restricted to: {AGENT_WORKDIR}
Shell commands start in that directory.
""".strip()


TOOLS = [
    {
        "type": "function",
        "name": "read_skill",
        "description": (
            "Load the complete SKILL.md for one available workspace skill. "
            "Call this before acting when a user names a skill or the task "
            "clearly matches one from the skills catalog."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                },
            },
            "required": [
                "name",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_plan",
        "description": (
            "Updates the task plan. Provide the complete ordered plan; "
            "at most one step can be in_progress."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "maxLength": MAX_TODO_EXPLANATION_CHARS,
                },
                "plan": {
                    "type": "array",
                    "maxItems": MAX_TODO_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "maxLength": MAX_TODO_STEP_CHARS,
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "shell",
        "description": (
            "Run a shell command in the coding workspace. "
            "Use for inspection, tests, builds, git, docker, "
            "and other CLI work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                },
                "yield_time_ms": {
                    "type": "integer",
                    "minimum": 250,
                    "maximum": 30000,
                    "description": (
                        "Wait before yielding a still-running command as a "
                        "managed shell session. Defaults to 10000 ms; the "
                        "effective range is 250-30000 ms."
                    ),
                },
                "tty": {
                    "type": "boolean",
                    "description": (
                        "Allocate a PTY for an interactive command. False "
                        "or omitted uses plain pipes."
                    ),
                },
            },
            "required": [
                "command",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_stdin",
        "description": (
            "Write to or poll a managed shell session. Interactive input "
            "requires that shell was started with tty=true; empty chars poll. "
            "The control character \\u0003 interrupts the process group."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                },
                "chars": {
                    "type": "string",
                },
                "yield_time_ms": {
                    "type": "integer",
                    "minimum": 250,
                    "maximum": 300000,
                },
            },
            "required": [
                "session_id",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the coding workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5000,
                },
            },
            "required": [
                "path",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": (
            "Create or overwrite a UTF-8 text file in the "
            "coding workspace. Parent directories are created."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "content": {
                    "type": "string",
                },
            },
            "required": [
                "path",
                "content",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": (
            "Replace one exact text occurrence in a UTF-8 file. "
            "Fails if old_text is absent or appears multiple times "
            "unless replace_all=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "old_text": {
                    "type": "string",
                },
                "new_text": {
                    "type": "string",
                },
                "replace_all": {
                    "type": "boolean",
                },
            },
            "required": [
                "path",
                "old_text",
                "new_text",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_dir",
        "description": (
            "List files and directories in the coding workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
                "max_entries": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_goal",
        "description": (
            "Read the durable root-session goal. "
            "Returns null when no goal exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_goal",
        "description": (
            "Create a persistent autonomous goal only when "
            "the current user explicitly asks to establish one. "
            "Do not create goals merely because a task is "
            "complex or long-running. Do not call this tool "
            "when the user declines a persistent goal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                },
            },
            "required": [
                "objective",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_goal",
        "description": (
            "Update the durable root-session goal status. "
            "Use status=complete only after the full objective "
            "is achieved and verified. Use status=blocked when "
            "outside intervention is required. Always pass the "
            "current goal_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "string",
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "complete",
                        "blocked",
                    ],
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Required when status is blocked."
                    ),
                },
            },
            "required": [
                "goal_id",
                "status",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "chat_history",
        "description": (
            "Inspect Raptor's archived lossless chat transcripts when "
            "context needed for the current task is no longer present in the "
            "active checkpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "search",
                        "read",
                    ],
                },
                "query": {
                    "type": "string",
                },
                "session_id": {
                    "type": "string",
                },
                "all_sessions": {
                    "type": "boolean",
                },
                "start_seq": {
                    "type": "integer",
                    "minimum": 1,
                },
                "end_seq": {
                    "type": "integer",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": [
                "action",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "subagent",
        "description": (
            "List, start, or continue durable subagent threads using the "
            "configured subagent Responses host and model. Call without "
            "arguments to list threads, agent_id alone to inspect stored "
            "tasks/tool results/status, task to start a new thread, or task "
            "plus agent_id to continue one. If that thread is running, the "
            "task steers it at the next safe model boundary. Use only when "
            "the user explicitly requests delegation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                },
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Existing subagent thread to inspect, continue, or "
                        "steer while running."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Return immediately and notify the parent agent "
                        "when complete."
                    ),
                },
                "allow_subagents": {
                    "type": "boolean",
                    "description": (
                        "Allow this child to delegate another level. "
                        "Set only when the user explicitly requests nesting."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
]
