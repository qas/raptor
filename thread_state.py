"""Queries over the active temporary conversation thread."""

from typing import Any

import session


def current_thread() -> dict[str, Any] | None:
    value = session.state.get("thread")
    return value if isinstance(value, dict) else None


def thread_active() -> bool:
    thread = current_thread()
    return bool(
        thread
        and thread.get("id")
        and thread.get("session_id")
        and thread.get("parent_session_id")
    )


def thread_owner(thread: dict[str, Any] | None = None) -> str | None:
    value = thread or current_thread()
    if not value or not value.get("id"):
        return None
    return "thread:" + str(value["id"])
