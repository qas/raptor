"""Persistent state and process-wide runtime objects."""
import asyncio
import copy
import json
import secrets
import time
from typing import Any

from chat_provider import ConversationId

from chat_store import (
    create_session,
    ensure_chat_dirs,
    repair_all_chat_files,
    session_exists,
)
from config import (
    MAX_PENDING_STEERS,
    MAX_SUBAGENT_PENDING_INPUTS,
    MAX_SUBAGENT_RECORDS,
    RESPONSES_MODEL,
    STATE_PATH,
)
from runtime_events import RuntimeEvent
from todos import normalize_persisted_plan
from storage import write_text_atomic

DEFAULT_STATE: dict[str, Any] = {
    "model": None,
    "current_session_id": None,
    "todos": [],
    "approval_mode": "off",
    "pending_inputs": [],
    "active_root_turn": None,
    "interrupted_subagents": [],
    "subagents": {},
    "runtime": {},
    "goal": None,
    "thread": None,
}


def _prune_subagent_mapping(
    records: dict[str, Any],
    interrupted: list[Any],
) -> int:
    protected_ids = {
        str(item.get("id"))
        for item in interrupted
        if isinstance(item, dict) and item.get("id") is not None
    }
    protected_ids.update(
        str(agent_id)
        for agent_id, record in records.items()
        if isinstance(record, dict)
        and (
            record.get("status") == "running"
            or record.get("completion_pending")
        )
    )
    removable = sorted(
        (
            (str(agent_id), record)
            for agent_id, record in records.items()
            if str(agent_id) not in protected_ids
            and isinstance(record, dict)
        ),
        key=lambda pair: (
            float(pair[1].get("completed_at") or 0),
            float(pair[1].get("started_at") or 0),
        ),
        reverse=True,
    )
    keep_removable = MAX_SUBAGENT_RECORDS
    remove_ids = {
        agent_id for agent_id, _record in removable[keep_removable:]
    }
    for agent_id in remove_ids:
        records.pop(agent_id, None)
    return len(remove_ids)


def bounded_interrupted_subagents(items: list[Any]) -> list[dict[str, Any]]:
    """Retain the newest unique interrupted-subagent checkpoints."""
    newest_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        agent_id = str(item["id"])
        current = newest_by_id.get(agent_id)
        if current is None or float(item.get("interrupted_at") or 0) >= float(
            current.get("interrupted_at") or 0
        ):
            newest_by_id[agent_id] = item
    return sorted(
        newest_by_id.values(),
        key=lambda item: float(item.get("interrupted_at") or 0),
        reverse=True,
    )[:MAX_SUBAGENT_RECORDS]


def _ensure_session(result: dict[str, Any]) -> None:
    ensure_chat_dirs()
    session_id = result.get("current_session_id")
    if session_id and session_exists(str(session_id)):
        return
    result["current_session_id"] = create_session(kind="main")


def load_state() -> dict[str, Any]:
    result = copy.deepcopy(DEFAULT_STATE)
    if STATE_PATH.exists():
        try:
            loaded = json.loads(STATE_PATH.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not load state: {STATE_PATH}") from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(f"State root must be an object: {STATE_PATH}")
        for key in DEFAULT_STATE:
            if key in loaded:
                result[key] = loaded[key]
    if RESPONSES_MODEL:
        result["model"] = RESPONSES_MODEL
    if not isinstance(result.get("runtime"), dict):
        result["runtime"] = {}
    if not isinstance(result.get("subagents"), dict):
        result["subagents"] = {}
    interrupted_subagents = result.get("interrupted_subagents")
    result["interrupted_subagents"] = bounded_interrupted_subagents(
        interrupted_subagents if isinstance(interrupted_subagents, list) else []
    )
    pending_inputs = result.get("pending_inputs")
    if isinstance(pending_inputs, list):
        result["pending_inputs"] = [
            str(item)
            for item in pending_inputs[:MAX_PENDING_STEERS]
            if str(item)
        ]
    else:
        result["pending_inputs"] = []
    result["todos"] = normalize_persisted_plan(result.get("todos"))
    goal = result.get("goal")
    if goal is not None and not isinstance(goal, dict):
        result["goal"] = None
    if isinstance(result.get("goal"), dict):
        result["goal"].setdefault("blocked_reason", None)
        result["goal"].setdefault("notified_status", None)
        result["goal"]["todos"] = normalize_persisted_plan(
            result["goal"].get("todos")
        )
    thread = result.get("thread")
    if thread is not None and not isinstance(thread, dict):
        result["thread"] = None
    if isinstance(result.get("thread"), dict):
        thread = result["thread"]
        parent_id = str(thread.get("parent_session_id") or "")
        branch_id = str(thread.get("session_id") or "")
        if not session_exists(parent_id) or not session_exists(branch_id):
            parent_interrupted_subagents = thread.get(
                "parent_interrupted_subagents"
            )
            result["thread"] = None
            if session_exists(parent_id):
                result["current_session_id"] = parent_id
                result["interrupted_subagents"] = (
                    parent_interrupted_subagents
                    if isinstance(parent_interrupted_subagents, list)
                    else []
                )
        else:
            result["current_session_id"] = branch_id
    for record in result["subagents"].values():
        if not isinstance(record, dict):
            continue
        record["todos"] = normalize_persisted_plan(record.get("todos"))
        pending_subagent_inputs = record.get("pending_inputs")
        record["pending_inputs"] = (
            [
                str(item)
                for item in pending_subagent_inputs[
                    :MAX_SUBAGENT_PENDING_INPUTS
                ]
                if str(item)
            ]
            if isinstance(pending_subagent_inputs, list)
            else []
        )
        record.setdefault("recovery_context", None)
        record.setdefault("completion_pending", False)
        record.setdefault("completion_notified_at", None)
        record.setdefault("completion_attempts", 0)
        record.setdefault("parent_session_id", None)
        record["task_count"] = max(1, int(record.get("task_count") or 1))
        if record.get("status") == "running":
            record["pending_inputs"] = []
            record["status"] = "interrupted"
            record["error"] = (
                "Process exited while subagent was running"
            )
            record["completed_at"] = int(time.time())
            checkpoint = {
                "id": record.get("id"),
                "session_id": record.get("session_id"),
                "interrupted_at": time.time(),
                "tool_events": list(
                    record.get("tool_events") or []
                ),
            }
            result["interrupted_subagents"] = [
                item
                for item in result["interrupted_subagents"]
                if item.get("id") != checkpoint.get("id")
            ]
            result["interrupted_subagents"].append(checkpoint)
    result["interrupted_subagents"] = bounded_interrupted_subagents(
        result["interrupted_subagents"]
    )
    _prune_subagent_mapping(
        result["subagents"],
        result["interrupted_subagents"],
    )
    if result.get("approval_mode") not in {"on", "off"}:
        result["approval_mode"] = "off"
    return result


state = load_state()

# Chat presentation only — never persisted in durable state.
goal_pin_message_id: int | str | None = None
goal_pin_goal_id: str | None = None
# Runtime-only pinned chat status slot. Approval, a temporary thread, and goals
# project into this one message; steering remains standalone and transient.
pinned_status_conversation_id: int | str | None = None
pinned_status_message_id: int | str | None = None
pinned_status_owner: str | None = None
# Runtime-only: True only for the current user turn when it explicitly
# requested persistent goal creation. Never persisted.
goal_creation_authorized: bool = False

steer_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
pending_steers: dict[str, dict[str, Any]] = {}

runtime_event_queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()

subagent_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
subagent_tasks: dict[str, asyncio.Task] = {}
subagent_records: dict[str, dict[str, Any]] = state["subagents"]

pending_approvals: dict[str, dict[str, Any]] = {}

APPROVAL_TOOLS = {
    "shell",
    "write_file",
    "edit_file",
}

DAEMON_MODE = False

responses: Any


def save_state() -> None:
    write_text_atomic(
        STATE_PATH,
        json.dumps(state, indent=2, ensure_ascii=False),
        mode=0o600,
    )


def prune_subagent_records() -> int:
    return _prune_subagent_mapping(
        state["subagents"],
        state.get("interrupted_subagents", []),
    )


def rehydrate_pending_inputs(chat_id: ConversationId) -> int:
    """Restore persisted steering texts into the in-memory steer queue.

    pending_inputs survive process exit; steer_queue does not. Call once at
    startup so queued steers are not lost.
    """
    pending = state.get("pending_inputs")
    if not isinstance(pending, list) or not pending:
        return 0
    count = 0
    for text in pending:
        content = str(text)
        if not content:
            continue
        steer_id = secrets.token_hex(4)
        entry = {
            "id": steer_id,
            "chat_id": chat_id,
            "text": content,
            "message_id": None,
            "status": "queued",
            "rehydrated": True,
        }
        pending_steers[steer_id] = entry
        steer_queue.put_nowait(entry)
        count += 1
    return count


def bootstrap_runtime_storage() -> dict[str, int]:
    """Repair chat JSONL tails and ensure directories exist."""
    ensure_chat_dirs()
    _ensure_session(state)
    repaired = repair_all_chat_files()
    return {"repaired_chats": repaired}
