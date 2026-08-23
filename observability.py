"""Structured runtime events and user-visible activity labels."""

import json
import re
import time
from typing import Any


_REDACTED = "[REDACTED]"
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
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "source": source,
        "event": event,
    }
    if data is not None:
        record["data"] = data
    print(
        json.dumps(redact_sensitive(record), ensure_ascii=False, default=str),
        flush=True,
    )
