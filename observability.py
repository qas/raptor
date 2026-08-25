"""Structured runtime events and user-visible activity labels."""

import json
import re
import time
from typing import Any

from config import MAX_TOOL_OUTPUT


_REDACTED = "[REDACTED]"
_MAX_LOG_STRING = 2_000
_MAX_LOG_COLLECTION = 50
_MAX_LOG_DEPTH = 6
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(api\.telegram\.org/bot)[^/\s]+"),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)"
        r"[^&\s]+"
    ),
    re.compile(
        r'(?i)(["\'](?:api[_-]?key|access[_-]?token|token|secret|password)'
        r'["\']\s*:\s*["\'])[^"\']+'
    ),
    re.compile(
        r"(?i)((?<![A-Za-z0-9_])(?:[A-Za-z0-9_-]*"
        r"(?:api[-_]?key|access[-_]?token|bot[-_]?token|secret|password)"
        r"|authorization|token)\s*[:=]\s*(?:bearer\s+)?)"
        r"[^\s,;\"']+"
    ),
)


def _sensitive_key(value: object) -> bool:
    key = str(value).strip().lower().replace("-", "_")
    return key in {
        "authorization",
        "api_key",
        "access_token",
        "token",
        "secret",
        "password",
    } or key.endswith(
        ("_api_key", "_access_token", "_token", "_secret", "_password")
    )


def redact_sensitive(value: Any) -> Any:
    """Return a log-safe copy with common credential forms removed."""
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_TEXT_PATTERNS:
            redacted = pattern.sub(lambda match: match.group(1) + _REDACTED, redacted)
        return redacted
    if isinstance(value, dict):
        return {
            key: _REDACTED if _sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value


def bound_log_value(value: Any, *, depth: int = 0) -> Any:
    """Bound diagnostic payloads before serialization."""
    if depth >= _MAX_LOG_DEPTH:
        return "[MAX_DEPTH]"
    if isinstance(value, str):
        if len(value) <= _MAX_LOG_STRING:
            return value
        omitted = len(value) - _MAX_LOG_STRING
        return value[:_MAX_LOG_STRING] + f"... [{omitted} chars omitted]"
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            key: bound_log_value(item, depth=depth + 1)
            for key, item in items[:_MAX_LOG_COLLECTION]
        }
        if len(items) > _MAX_LOG_COLLECTION:
            bounded["_omitted_fields"] = len(items) - _MAX_LOG_COLLECTION
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_items = [
            bound_log_value(item, depth=depth + 1)
            for item in value[:_MAX_LOG_COLLECTION]
        ]
        if len(value) > _MAX_LOG_COLLECTION:
            bounded_items.append(
                f"[{len(value) - _MAX_LOG_COLLECTION} items omitted]"
            )
        return bounded_items
    return value


def tool_activity(call: dict[str, Any]) -> str:
    activities = {
        "update_plan": "updating todos",
        "shell": "running a shell command",
        "write_stdin": "checking a shell session",
        "cancel": "stopping background work",
        "read_file": "reading a file",
        "read_skill": "loading a skill",
        "write_file": "writing a file",
        "edit_file": "editing a file",
        "list_dir": "listing a directory",
        "get_goal": "reading the goal",
        "update_goal": "updating the goal",
        "set_goal": "setting a goal",
        "chat_history": "searching chat history",
        "subagent": "delegating to a subagent",
    }
    name = str(call.get("name") or "unknown tool")
    return activities.get(name, f"using {name}")


def log_agent_activity(activity: str) -> None:
    log_event("agent", "activity", {"activity": activity})


def log_exception(
    source: str,
    event: str,
    exc: BaseException,
    data: dict[str, Any] | None = None,
) -> None:
    payload = dict(data or {})
    payload.update({"type": type(exc).__name__, "message": str(exc)})
    log_event(source, event, payload)


def log_event(source: str, event: str, data: Any = None) -> None:
    """Write a bounded diagnostic event with credential redaction."""
    _write_record(source, event, data, audit=False)


def log_shell_start(
    *,
    session_id: str,
    command: str,
    chat_id: object,
    parent_session_id: str | None,
    pid: int,
    timeout: int | None,
    tty: bool,
) -> None:
    """Record authorization for one bounded, exact shell command."""
    if len(command) > MAX_TOOL_OUTPUT:
        raise ValueError(
            f"shell audit command exceeds {MAX_TOOL_OUTPUT} characters"
        )
    _write_record(
        "shell",
        "authorized",
        {
            "session_id": session_id,
            "command": command,
            "chat_id": chat_id,
            "parent_session_id": parent_session_id,
            "pid": pid,
            "timeout": timeout,
            "tty": tty,
        },
        audit=True,
    )


def _write_record(
    source: str,
    event: str,
    data: Any,
    *,
    audit: bool,
) -> None:
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "source": source,
        "event": event,
    }
    if audit:
        record["audit"] = True
    if data is not None:
        record["data"] = data
    serialized = record if audit else bound_log_value(redact_sensitive(record))
    print(
        json.dumps(
            serialized,
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
