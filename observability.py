"""Structured runtime events and user-visible activity labels."""

import json
import time
from typing import Any


def tool_activity(call: dict[str, Any]) -> str:
    activities = {
        "update_plan": "updating todos",
        "shell": "running a shell command",
        "write_stdin": "checking a shell session",
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
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
