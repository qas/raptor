"""Configuration and tool schemas."""
import ipaddress
import math
import os
import re
from urllib.parse import urlsplit

from raptor.config_document import config_section, load_config_document
from raptor.shell.filesystem_permissions import (
    DEFAULT_GLOB_SCAN_MAX_DEPTH,
    FileAccessPolicy,
)
from raptor.runtime_paths import AGENT_WORKDIR, CHAT_DIR, LOG_PATH, RAPTOR_HOME, STATE_PATH

from raptor.agent.todos import (
    MAX_TODO_EXPLANATION_CHARS,
    MAX_TODO_ITEMS,
    MAX_TODO_STEP_CHARS,
)


def _env_int(
    name: str,
    default: int,
    *,
    configured: object = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    from_environment = name in os.environ
    raw: object = os.environ[name] if from_environment else configured
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer")
    if from_environment:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    elif isinstance(raw, int):
        value = raw
    else:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    configured: object = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    from_environment = name in os.environ
    raw: object = os.environ[name] if from_environment else configured
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a number")
    if from_environment:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _env_bool(
    name: str,
    default: bool,
    *,
    configured: object = None,
) -> bool:
    from_environment = name in os.environ
    raw: object = os.environ[name] if from_environment else configured
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        return raw
    if not from_environment:
        raise ValueError(f"{name} must be a boolean")
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a boolean")
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int_tuple(
    name: str,
    default: tuple[int, ...],
    *,
    configured: object = None,
) -> tuple[int, ...]:
    from_environment = name in os.environ
    raw: object = os.environ[name] if from_environment else configured
    if raw is None:
        return default
    if from_environment and isinstance(raw, str):
        parts = raw.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError(f"{name} must contain comma-separated integers")
        try:
            values = tuple(int(part.strip()) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"{name} must contain comma-separated integers"
            ) from exc
    elif not from_environment and isinstance(raw, list) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw
    ):
        values = tuple(raw)
    else:
        raise ValueError(f"{name} must contain integers")
    if any(value == 0 for value in values):
        raise ValueError(f"{name} entries must not be zero")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values


def _env_proxy(name: str, *, configured: object = None) -> str | None:
    raw_value: object = os.environ[name] if name in os.environ else configured
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"{name} must be a string")
    raw = raw_value.strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https", "socks5h"}:
        raise ValueError(
            f"{name} must use http, https, or socks5h"
        )
    if not parsed.hostname:
        raise ValueError(f"{name} must include a hostname")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not include a path, query, or fragment")
    return raw


_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _proxy_bypass_host(name: str, raw: str, *, wildcard: bool) -> str:
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        try:
            host = raw.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError(f"{name} contains an invalid hostname") from exc
        labels = host.split(".")
        if (
            len(host) > 253
            or any(not _HOST_LABEL.fullmatch(label) for label in labels)
        ):
            raise ValueError(f"{name} contains an invalid hostname")
        return host
    if wildcard:
        raise ValueError(f"{name} cannot wildcard an IP address")
    return address.compressed


def _env_proxy_bypass(
    name: str,
    *,
    configured: object = None,
) -> tuple[str, ...]:
    from_environment = name in os.environ
    raw_value: object = os.environ[name] if from_environment else configured
    if raw_value is None:
        return ()
    if from_environment and isinstance(raw_value, str):
        if not raw_value.strip():
            return ()
        parts = raw_value.split(",")
    elif not from_environment and isinstance(raw_value, list) and all(
        isinstance(part, str) for part in raw_value
    ):
        parts = raw_value
    else:
        raise ValueError(f"{name} must contain hosts")
    if any(not part.strip() for part in parts):
        raise ValueError(f"{name} must contain comma-separated hosts")
    entries = []
    for part in parts:
        entry = part.strip()
        wildcard = entry.startswith("*.")
        if "*" in entry[2:] or ("*" in entry and not wildcard):
            raise ValueError(f"{name} only supports leading *. wildcards")
        host = _proxy_bypass_host(
            name,
            entry[2:] if wildcard else entry,
            wildcard=wildcard,
        )
        entries.append(f"*.{host}" if wildcard else host)
    if len(set(entries)) != len(entries):
        raise ValueError(f"{name} entries must be unique")
    return tuple(entries)


def _env_string(
    name: str,
    default: str,
    *,
    configured: object = None,
    allow_empty: bool = False,
) -> str:
    raw: object = os.environ[name] if name in os.environ else configured
    if raw is None:
        raw = default
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string")
    value = raw.strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return value


def _env_string_tuple(
    name: str,
    default: tuple[str, ...],
    *,
    configured: object = None,
) -> tuple[str, ...]:
    from_environment = name in os.environ
    raw: object = os.environ[name] if from_environment else configured
    if raw is None:
        values = default
    elif from_environment and isinstance(raw, str):
        values = tuple(part.strip() for part in raw.split(","))
    elif not from_environment and isinstance(raw, list) and all(
        isinstance(part, str) for part in raw
    ):
        values = tuple(part.strip() for part in raw)
    else:
        raise ValueError(f"{name} must contain strings")
    if not values or any(not value for value in values):
        raise ValueError(f"{name} must contain non-empty entries")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} entries must be unique")
    return values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG = load_config_document()
_NETWORK = config_section(_CONFIG, "network", {"proxy", "no_proxy"})
_PERMISSIONS = config_section(_CONFIG, "permissions", {"filesystem"})
_FILESYSTEM = config_section(
    _PERMISSIONS,
    "filesystem",
    {"deny_read", "glob_scan_max_depth"},
)
_CHAT = config_section(
    _CONFIG,
    "chat",
    {
        "providers",
        "streaming",
        "stream_interval",
        "tool_activity",
        "max_pending_steers",
        "max_runtimes",
    },
)
_TELEGRAM = config_section(
    _CONFIG,
    "telegram",
    {
        "user_id",
        "chat_ids",
        "max_retries",
        "markdown",
        "subagent_topics_silent",
    },
)
_RESPONSES_SERVER = config_section(
    _CONFIG,
    "responses_server",
    {
        "host",
        "port",
        "max_body",
        "max_connections",
        "max_pending",
        "max_status_messages",
        "max_stream_events",
        "read_timeout",
    },
)
_SUBAGENTS = config_section(
    _CONFIG,
    "subagents",
    {
        "max_depth",
        "max_records",
        "max_tool_events",
        "max_pending_inputs",
        "max_background",
    },
)
_TOOLS_CONFIG = config_section(
    _CONFIG,
    "tools",
    {"max_rounds", "max_output"},
)
_SHELL = config_section(_CONFIG, "shell", {"timeout"})
_STATE = config_section(_CONFIG, "state", {"max_load_bytes"})
_COMPACTION = config_section(
    _CONFIG,
    "compaction",
    {
        "model_provider",
        "model",
        "output_tokens",
        "generation_tokens",
        "reasoning_effort",
        "keep_recent_tokens",
        "user_anchor_tokens",
        "max_record_chars",
        "context_ratio",
        "context_safety_tokens",
    },
)

RAPTOR_PROXY = _env_proxy("RAPTOR_PROXY", configured=_NETWORK.get("proxy"))
if "RAPTOR_PROXY" not in os.environ and RAPTOR_PROXY is not None:
    configured_proxy = urlsplit(RAPTOR_PROXY)
    if (
        configured_proxy.username is not None
        or configured_proxy.password is not None
    ):
        raise ValueError(
            "network.proxy must not contain credentials; use RAPTOR_PROXY"
        )
RAPTOR_NO_PROXY = _env_proxy_bypass(
    "RAPTOR_NO_PROXY",
    configured=_NETWORK.get("no_proxy"),
)
if RAPTOR_NO_PROXY and RAPTOR_PROXY is None:
    raise ValueError("RAPTOR_NO_PROXY requires RAPTOR_PROXY")

CHAT_PROVIDERS = _env_string_tuple(
    "CHAT_PROVIDERS",
    ("telegram", "responses_api"),
    configured=_CHAT.get("providers"),
)

# Inbound, OpenAI Responses-compatible chat provider. This is independent from
# outbound model providers configured in ``.raptor/config.toml``.
RESPONSES_SERVER_HOST = _env_string(
    "RESPONSES_SERVER_HOST",
    "127.0.0.1",
    configured=_RESPONSES_SERVER.get("host"),
)
RESPONSES_SERVER_PORT = _env_int(
    "RESPONSES_SERVER_PORT",
    8787,
    configured=_RESPONSES_SERVER.get("port"),
    minimum=1,
    maximum=65535,
)
RESPONSES_SERVER_API_KEY = os.getenv(
    "RESPONSES_SERVER_API_KEY",
    "",
).strip()
RESPONSES_SERVER_MAX_BODY = _env_int(
    "RESPONSES_SERVER_MAX_BODY",
    1_048_576,
    configured=_RESPONSES_SERVER.get("max_body"),
    minimum=1024,
)
RESPONSES_SERVER_MAX_CONNECTIONS = _env_int(
    "RESPONSES_SERVER_MAX_CONNECTIONS",
    128,
    configured=_RESPONSES_SERVER.get("max_connections"),
    minimum=1,
)
RESPONSES_SERVER_MAX_PENDING = _env_int(
    "RESPONSES_SERVER_MAX_PENDING",
    64,
    configured=_RESPONSES_SERVER.get("max_pending"),
    minimum=1,
)
RESPONSES_SERVER_MAX_STATUS_MESSAGES = _env_int(
    "RESPONSES_SERVER_MAX_STATUS_MESSAGES",
    256,
    configured=_RESPONSES_SERVER.get("max_status_messages"),
    minimum=1,
)
RESPONSES_SERVER_MAX_STREAM_EVENTS = _env_int(
    "RESPONSES_SERVER_MAX_STREAM_EVENTS",
    256,
    configured=_RESPONSES_SERVER.get("max_stream_events"),
    minimum=4,
)
RESPONSES_SERVER_READ_TIMEOUT = _env_float(
    "RESPONSES_SERVER_READ_TIMEOUT",
    10.0,
    configured=_RESPONSES_SERVER.get("read_timeout"),
    minimum=0.1,
)

# Telegram adapter configuration. These are optional at framework import
# time and validated only when the Telegram provider is initialized.
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_USER_ID = _env_int(
    "TG_USER_ID",
    0,
    configured=_TELEGRAM.get("user_id"),
    minimum=0,
)
TG_CHAT_IDS = _env_int_tuple(
    "TG_CHAT_IDS",
    (TG_USER_ID,) if TG_USER_ID else (),
    configured=_TELEGRAM.get("chat_ids"),
)
TG_MAX_RETRIES = _env_int(
    "TG_MAX_RETRIES",
    3,
    configured=_TELEGRAM.get("max_retries"),
    minimum=0,
)

MAX_SUBAGENT_DEPTH = _env_int(
    "MAX_SUBAGENT_DEPTH",
    3,
    configured=_SUBAGENTS.get("max_depth"),
    minimum=1,
)

MAX_SUBAGENT_RECORDS = _env_int(
    "MAX_SUBAGENT_RECORDS",
    100,
    configured=_SUBAGENTS.get("max_records"),
    minimum=1,
)
MAX_SUBAGENT_TOOL_EVENTS = _env_int(
    "MAX_SUBAGENT_TOOL_EVENTS",
    500,
    configured=_SUBAGENTS.get("max_tool_events"),
    minimum=1,
)
MAX_SUBAGENT_PENDING_INPUTS = _env_int(
    "MAX_SUBAGENT_PENDING_INPUTS",
    64,
    configured=_SUBAGENTS.get("max_pending_inputs"),
    minimum=1,
)
MAX_BACKGROUND_SUBAGENTS = _env_int(
    "MAX_BACKGROUND_SUBAGENTS",
    16,
    configured=_SUBAGENTS.get("max_background"),
    minimum=1,
)

MAX_TOOL_ROUNDS = _env_int(
    "MAX_TOOL_ROUNDS",
    0,
    configured=_TOOLS_CONFIG.get("max_rounds"),
    minimum=0,
)
MAX_PENDING_STEERS = _env_int(
    "MAX_PENDING_STEERS",
    64,
    configured=_CHAT.get("max_pending_steers"),
    minimum=1,
)
MAX_CHAT_RUNTIMES = _env_int(
    "MAX_CHAT_RUNTIMES",
    1024,
    configured=_CHAT.get("max_runtimes"),
    minimum=1,
)
MAX_STATE_LOAD_BYTES = _env_int(
    "MAX_STATE_LOAD_BYTES",
    16 * 1024 * 1024,
    configured=_STATE.get("max_load_bytes"),
    minimum=1024,
)

SHELL_TIMEOUT = _env_int(
    "SHELL_TIMEOUT",
    0,
    configured=_SHELL.get("timeout"),
    minimum=0,
)

FILESYSTEM_POLICY = FileAccessPolicy.create(
    AGENT_WORKDIR,
    _FILESYSTEM.get("deny_read", []),
    _FILESYSTEM.get("glob_scan_max_depth", DEFAULT_GLOB_SCAN_MAX_DEPTH),
)

MAX_TOOL_OUTPUT = _env_int(
    "MAX_TOOL_OUTPUT",
    30_000,
    configured=_TOOLS_CONFIG.get("max_output"),
    minimum=1024,
)

COMPACTION_OUTPUT_TOKENS = _env_int(
    "COMPACTION_OUTPUT_TOKENS",
    4_000,
    configured=_COMPACTION.get("output_tokens"),
    minimum=256,
)

# The Responses API counts hidden reasoning and visible checkpoint text against
# the same generation limit.  Keep the durable checkpoint cap independent from
# the model's generation allowance so a reasoning model cannot consume the
# entire budget before emitting its summary.
COMPACTION_GENERATION_TOKENS = _env_int(
    "COMPACTION_GENERATION_TOKENS",
    COMPACTION_OUTPUT_TOKENS + 4096,
    configured=_COMPACTION.get("generation_tokens"),
    minimum=COMPACTION_OUTPUT_TOKENS,
)

COMPACTION_REASONING_EFFORT = _env_string(
    "COMPACTION_REASONING_EFFORT",
    "low",
    configured=_COMPACTION.get("reasoning_effort"),
    allow_empty=True,
) or None

COMPACT_KEEP_RECENT_TOKENS = _env_int(
    "COMPACT_KEEP_RECENT_TOKENS",
    20_000,
    configured=_COMPACTION.get("keep_recent_tokens"),
    minimum=0,
)

# Original user requests retained alongside a generated checkpoint.  These are
# semantic anchors, not another model-written summary.
COMPACTION_USER_ANCHOR_TOKENS = _env_int(
    "COMPACTION_USER_ANCHOR_TOKENS",
    20_000,
    configured=_COMPACTION.get("user_anchor_tokens"),
    minimum=0,
)

# Per-record char cap when rendering archive items into a checkpoint
# request. Canonical JSONL is never truncated.
COMPACTION_MAX_RECORD_CHARS = _env_int(
    "COMPACTION_MAX_RECORD_CHARS",
    12_000,
    configured=_COMPACTION.get("max_record_chars"),
    minimum=1024,
)

CONTEXT_COMPACT_RATIO = _env_float(
    "CONTEXT_COMPACT_RATIO",
    0.82,
    configured=_COMPACTION.get("context_ratio"),
    minimum=0.50,
    maximum=0.95,
)

CONTEXT_SAFETY_TOKENS = _env_int(
    "CONTEXT_SAFETY_TOKENS",
    4096,
    configured=_COMPACTION.get("context_safety_tokens"),
    minimum=0,
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


def model_context_input_budget(model_context_tokens: int | None) -> int:
    return _context_input_budget(model_context_tokens or 0)


def _compaction_generation_budget(model_context_tokens: int) -> int:
    """Generation allowance that remains sane on smaller context models."""
    if not model_context_tokens:
        return COMPACTION_GENERATION_TOKENS
    return min(
        COMPACTION_GENERATION_TOKENS,
        max(1024, model_context_tokens // 4),
    )


def model_compaction_generation_budget(
    model_context_tokens: int | None,
) -> int:
    return _compaction_generation_budget(model_context_tokens or 0)


CHAT_STREAMING = _env_bool(
    "CHAT_STREAMING",
    True,
    configured=_CHAT.get("streaming"),
)

CHAT_TOOL_ACTIVITY = _env_bool(
    "CHAT_TOOL_ACTIVITY",
    True,
    configured=_CHAT.get("tool_activity"),
)

TELEGRAM_MARKDOWN = _env_bool(
    "TELEGRAM_MARKDOWN",
    True,
    configured=_TELEGRAM.get("markdown"),
)

# Telegram's disable_notification flag suppresses sound; it does not hide the
# notification or mute a topic in the user's client.
TELEGRAM_SUBAGENT_TOPICS_SILENT = _env_bool(
    "TELEGRAM_SUBAGENT_TOPICS_SILENT",
    True,
    configured=_TELEGRAM.get("subagent_topics_silent"),
)

CHAT_STREAM_INTERVAL = _env_float(
    "CHAT_STREAM_INTERVAL",
    0.35,
    configured=_CHAT.get("stream_interval"),
    minimum=0.01,
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

IDENTITY: AGENTS.md and MEMORY.md from the workspace root are included below
these base instructions. Internalize the identity, conventions, durable
context, and working style they define. Do not narrate them back to the user.

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
- After starting a background subagent, acknowledge the start and end your
  turn. Do not poll it or run wait/sleep commands; its completion starts a new
  parent turn automatically.
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
Configured deny-read paths are inaccessible to filesystem and shell tools.
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
                    "maxLength": MAX_TOOL_OUTPUT,
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Maximum runtime in seconds. Zero disables the "
                        "deadline; defaults to SHELL_TIMEOUT."
                    ),
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
                    "maxLength": MAX_TOOL_OUTPUT,
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
                    "maxLength": MAX_TOOL_OUTPUT,
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
                    "maxLength": MAX_TOOL_OUTPUT,
                },
                "new_text": {
                    "type": "string",
                    "maxLength": MAX_TOOL_OUTPUT,
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
            "Update the durable root-session goal objective or status. "
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
                "objective": {
                    "type": "string",
                    "description": (
                        "A revised objective when the active goal's scope or "
                        "wording genuinely changes."
                    ),
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
        "name": "cancel",
        "description": (
            "Cancel one running background resource by its exact identifier. "
            "Use kind=subagent with an agent_id returned by subagent, or "
            "kind=shell with a session_id returned by shell. This is targeted; "
            "/stop all cancels background work owned by the current chat."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "subagent",
                        "shell",
                    ],
                },
                "id": {
                    "type": "string",
                    "description": "Exact agent_id or shell session_id.",
                },
            },
            "required": [
                "kind",
                "id",
            ],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "subagent",
        "description": (
            "List, start, or continue durable subagent threads using the "
            "parent agent's model target or an explicitly selected configured "
            "model provider and model. Call without "
            "arguments for a compact running-first roster and authoritative "
            "counts; use the running count rather than counting returned "
            "rows when truncated is true. Use agent_id alone to inspect stored "
            "public status and final result, task to start a new thread, or "
            "task plus agent_id to continue one, or agent_id plus delete to "
            "delete a stopped thread. If that thread is running, the task "
            "steers it at the next safe model boundary. Use only when the "
            "user explicitly requests delegation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "maxLength": MAX_TOOL_OUTPUT,
                },
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Existing subagent thread to inspect, continue, or "
                        "steer while running."
                    ),
                },
                "model_provider": {
                    "type": "string",
                    "description": (
                        "Configured model-provider ID for a new subagent. "
                        "Omit to inherit the parent provider."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Model ID for a new subagent. Omit to inherit the "
                        "parent model, or use the selected provider's default."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Return immediately. A successful background start "
                        "automatically starts a new parent turn when the "
                        "subagent completes; do not poll or wait for it."
                    ),
                },
                "delete": {
                    "type": "boolean",
                    "description": (
                        "Delete a stopped subagent and its activity surface. "
                        "Requires agent_id and cannot be combined with task."
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
