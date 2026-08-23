"""Durable root-session goal lifecycle."""
import secrets
import time
from typing import Any

from chat_provider import ConversationId

import chat_runtime
import presentation
from session import save_state, state
import session
from observability import log_event, log_exception
from thread_state import thread_active
from thread_status import ensure_thread_status

GOAL_ACTIVE = "active"
GOAL_PAUSED = "paused"
GOAL_BLOCKED = "blocked"
GOAL_COMPLETE = "complete"

GOAL_STATUSES = {
    GOAL_ACTIVE,
    GOAL_PAUSED,
    GOAL_BLOCKED,
    GOAL_COMPLETE,
}

MAX_GOAL_CHARS = 8000
MAX_BLOCKED_REASON_CHARS = 2000


def _now() -> float:
    return time.time()


def create_goal(objective: str) -> dict[str, Any]:
    cleaned = " ".join(str(objective).split())
    if not cleaned:
        raise ValueError("goal objective is empty")
    if len(cleaned) > MAX_GOAL_CHARS:
        raise ValueError(
            f"goal objective exceeds {MAX_GOAL_CHARS} characters"
        )
    now = _now()
    return {
        "id": secrets.token_hex(8),
        "objective": cleaned,
        "status": GOAL_ACTIVE,
        "blocked_reason": None,
        "notified_status": None,
        "todos": [],
        "created_at": now,
        "updated_at": now,
    }


def current_goal() -> dict[str, Any] | None:
    goal = state.get("goal")
    if not isinstance(goal, dict):
        return None
    return goal


def current_goal_id() -> str | None:
    goal = current_goal()
    if not goal:
        return None
    goal_id = goal.get("id")
    return str(goal_id) if goal_id else None


def goal_is_active() -> bool:
    goal = current_goal()
    return bool(
        goal
        and goal.get("status") == GOAL_ACTIVE
    )


def _touch(goal: dict[str, Any]) -> None:
    goal["updated_at"] = _now()


def _log(event: str, goal: dict[str, Any] | None) -> None:
    payload: dict[str, Any] = {}
    if goal:
        payload["goal_id"] = goal.get("id")
        payload["status"] = goal.get("status")
    log_event("goal", event, payload)


def replace_goal(objective: str) -> dict[str, Any]:
    previous = current_goal()
    goal = create_goal(objective)
    state["goal"] = goal
    state["todos"] = []
    save_state()
    _log(
        "goal_replaced" if previous else "goal_created",
        goal,
    )
    return goal


def pause_goal() -> tuple[dict[str, Any] | None, bool]:
    """Pause an active/blocked goal.

    Returns `(goal, changed)`. `goal` is None when no goal exists.
    `changed` is True only when status transitioned to paused.
    """
    goal = current_goal()
    if not goal:
        return None, False
    status = goal.get("status")
    if status not in {
        GOAL_ACTIVE,
        GOAL_BLOCKED,
    }:
        return goal, False
    goal["status"] = GOAL_PAUSED
    _touch(goal)
    save_state()
    _log("goal_paused", goal)
    return goal, True


def resume_goal() -> dict[str, Any] | None:
    goal = current_goal()
    if not goal:
        return None
    status = goal.get("status")
    if status not in {
        GOAL_PAUSED,
        GOAL_BLOCKED,
    }:
        return goal
    goal["status"] = GOAL_ACTIVE
    goal["blocked_reason"] = None
    goal["notified_status"] = None
    _touch(goal)
    save_state()
    _log("goal_resumed", goal)
    return goal


def complete_goal(expected_goal_id: str) -> bool:
    goal = current_goal()
    if not goal or str(goal.get("id")) != str(
        expected_goal_id
    ):
        _log(
            "goal_stale_update",
            goal,
        )
        return False
    goal["status"] = GOAL_COMPLETE
    goal["blocked_reason"] = None
    _touch(goal)
    save_state()
    _log("goal_completed", goal)
    return True


def block_goal(
    expected_goal_id: str,
    reason: str,
) -> bool:
    goal = current_goal()
    if not goal or str(goal.get("id")) != str(
        expected_goal_id
    ):
        _log(
            "goal_stale_update",
            goal,
        )
        return False
    cleaned = " ".join(str(reason).split())
    if len(cleaned) > MAX_BLOCKED_REASON_CHARS:
        cleaned = cleaned[
            :MAX_BLOCKED_REASON_CHARS
        ]
    goal["status"] = GOAL_BLOCKED
    goal["blocked_reason"] = cleaned or "blocked"
    _touch(goal)
    save_state()
    _log("goal_blocked", goal)
    return True


def clear_goal() -> None:
    goal = current_goal()
    state["goal"] = None
    state["todos"] = []
    save_state()
    _log("goal_cleared", goal)


def goal_instructions() -> str:
    if thread_active():
        return ""
    goal = current_goal()
    if not goal:
        return ""
    if goal.get("status") != GOAL_ACTIVE:
        return ""
    plan = list(goal.get("todos") or [])
    checklist = ""
    if plan:
        marks = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        checklist = "\n\nCurrent execution checklist:\n" + "\n".join(
            f"{marks.get(item['status'], '[ ]')} {item['step']}"
            for item in plan
        )
    return f"""
Active persistent goal:

{goal["objective"]}

Goal ID: {goal["id"]}
{checklist}

Continue making concrete progress toward the full goal.
Do not narrow the goal merely to finish this turn.
Continue across turns until the requested end state is actually achieved.
For multi-step work, maintain the execution checklist with update_plan. Mark
each verified step completed promptly and keep at most one step in_progress.
The checklist supports execution; it does not determine goal completion.
When the entire goal is achieved and verified, call update_goal with
status="complete". If meaningful progress cannot continue without outside
intervention, call update_goal with status="blocked" and explain why.
Do not mark the goal complete merely because one intermediate task or subtask finished.
""".strip()


def todo_store_for_execution() -> dict[str, Any]:
    """Select the canonical checklist owner for the current agent turn."""
    goal = current_goal()
    if (
        not thread_active()
        and goal
        and goal.get("status") == GOAL_ACTIVE
    ):
        return goal
    return state


def todo_store_for_display() -> dict[str, Any]:
    """Show durable goal progress while a goal can still be resumed."""
    goal = current_goal()
    if (
        not thread_active()
        and goal
        and goal.get("status") in {
            GOAL_ACTIVE,
            GOAL_PAUSED,
            GOAL_BLOCKED,
        }
    ):
        return goal
    return state


def goal_continuation_input() -> str:
    return (
        "Continue working toward the active persistent goal. "
        "Review current workspace and durable state, then make "
        "the next concrete progress toward completion."
    )


def combine_instructions(*parts: str) -> str:
    return "\n\n".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )


def format_goal_status() -> str:
    goal = current_goal()
    if not goal:
        return "goal: none"
    status = str(goal.get("status") or "none")
    objective = str(goal.get("objective") or "")
    if len(objective) > 120:
        objective = objective[:117] + "..."
    lines = [f"goal: {status}"]
    if objective:
        lines.append(f"goal objective: {objective}")
    if (
        status == GOAL_BLOCKED
        and goal.get("blocked_reason")
    ):
        reason = str(goal["blocked_reason"])
        if len(reason) > 160:
            reason = reason[:157] + "..."
        lines.append(f"goal blocked: {reason}")
    return "\n".join(lines)


def goal_pin_text(goal: dict[str, Any]) -> str:
    objective = str(goal.get("objective") or "")
    if len(objective) > 120:
        objective = objective[:117] + "..."
    status = str(goal.get("status") or "unknown")
    return f"Goal {status}: {objective}"


def _higher_priority_pin_active(
    chat_id: ConversationId,
    *,
    ignore_owner: str | None = None,
) -> bool:
    """True when approval owns the pinned status slot."""
    pinned_owner = session.pinned_status_owner
    if (
        session.pinned_status_conversation_id == chat_id
        and pinned_owner != ignore_owner
        and isinstance(pinned_owner, str)
        and pinned_owner.startswith("approval:")
    ):
        return True
    for entry in session.pending_approvals.values():
        if entry.get("chat_id") != chat_id:
            continue
        if entry.get("ui_finalized"):
            continue
        if entry.get("message_id") is not None:
            return True
    return False


async def remove_goal_pin(chat_id: ConversationId) -> None:
    """Clear goal tracking, then remove its persistent chat status."""
    message_id = session.goal_pin_message_id
    goal_id = session.goal_pin_goal_id or ""
    session.goal_pin_message_id = None
    session.goal_pin_goal_id = None
    if message_id is not None:
        try:
            await presentation.clear_goal_pin(
                chat_id,
                message_id,
                goal_id,
            )
        except Exception as exc:
            log_exception("goal", "goal_pin_remove_error", exc)


async def suspend_goal_pin(chat_id: ConversationId) -> None:
    """Release goal ownership without changing durable goal state."""
    await remove_goal_pin(chat_id)


async def ensure_goal_pin(
    chat_id: ConversationId,
    *,
    replace_owner: str | None = None,
) -> None:
    """Ensure exactly one pinned status exists for the active goal."""
    if thread_active():
        return
    if _higher_priority_pin_active(
        chat_id,
        ignore_owner=replace_owner,
    ):
        return
    goal = current_goal()
    if not goal or goal.get("status") != GOAL_ACTIVE:
        await remove_goal_pin(chat_id)
        return
    goal_id = str(goal.get("id") or "")
    if (
        session.goal_pin_message_id is not None
        and session.goal_pin_goal_id == goal_id
    ):
        return
    await remove_goal_pin(chat_id)
    text = goal_pin_text(goal)
    try:
        message_id = await presentation.show_goal_pin(
            chat_id,
            text,
            goal_id,
        )
    except Exception as exc:
        log_event(
            "goal",
            "goal_pin_error",
            {
                "goal_id": goal_id,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        return
    current = current_goal()
    if (
        not current
        or str(current.get("id") or "") != goal_id
        or current.get("status") != GOAL_ACTIVE
        or _higher_priority_pin_active(
            chat_id,
            ignore_owner=replace_owner,
        )
    ):
        try:
            await presentation.clear_goal_pin(
                chat_id,
                message_id,
                goal_id,
            )
        except Exception as exc:
            log_exception("goal", "stale_goal_pin_cleanup_error", exc)
        return
    session.goal_pin_message_id = message_id
    session.goal_pin_goal_id = goal_id


async def sync_goal_pin(
    chat_id: ConversationId,
    *,
    released_owner: str | None = None,
) -> None:
    """Re-project the pin from durable goal state and slot priority."""
    if _higher_priority_pin_active(
        chat_id,
        ignore_owner=released_owner,
    ):
        await suspend_goal_pin(chat_id)
        return
    if thread_active():
        await ensure_thread_status(
            chat_id,
            replace_owner=released_owner,
        )
        return
    goal = current_goal()
    if goal and goal.get("status") == GOAL_ACTIVE:
        await ensure_goal_pin(
            chat_id,
            replace_owner=released_owner,
        )
    elif released_owner:
        await presentation.clear_pinned_status(
            chat_id,
            owner=released_owner,
        )
    else:
        await remove_goal_pin(chat_id)


def prepare_goal_on_startup(*, root_interrupted: bool = False) -> str | None:
    """Preserve goal identity; pause active goals after unclean interruption."""
    goal = current_goal()
    if not goal:
        return None
    status = goal.get("status")
    if status not in GOAL_STATUSES:
        raise RuntimeError(f"Invalid persisted goal status: {status!r}")
    if thread_active():
        return None
    unclean = root_interrupted or bool(state.get("interrupted_subagents"))
    if status == GOAL_ACTIVE and unclean:
        goal["status"] = GOAL_PAUSED
        _touch(goal)
        save_state()
        _log("goal_paused", goal)
        return (
            "Active goal was paused after restart. "
            "/goal resume to continue."
        )
    return None


def get_goal_tool_result() -> dict[str, Any]:
    if thread_active():
        return {
            "ok": True,
            "goal": None,
            "thread_isolated": True,
        }
    goal = current_goal()
    if not goal:
        return {
            "ok": True,
            "goal": None,
        }
    return {
        "ok": True,
        "goal": {
            "id": goal.get("id"),
            "objective": goal.get("objective"),
            "status": goal.get("status"),
            "blocked_reason": goal.get(
                "blocked_reason"
            ),
        },
    }


def update_goal_tool(args: dict[str, Any]) -> dict[str, Any]:
    if thread_active():
        return {
            "ok": False,
            "error": "persistent goals are unavailable inside a thread",
        }
    goal_id = str(args.get("goal_id") or "").strip()
    status = str(args.get("status") or "").strip()
    objective = " ".join(str(args.get("objective") or "").split())
    reason = " ".join(str(args.get("reason") or "").split())
    if not goal_id:
        return {
            "ok": False,
            "error": "goal_id is required",
        }
    if not status and not objective:
        return {
            "ok": False,
            "error": "objective or status is required",
        }
    if status and status not in {GOAL_COMPLETE, GOAL_BLOCKED}:
        return {
            "ok": False,
            "error": (
                'status must be "complete" or "blocked"'
            ),
        }
    if status == GOAL_BLOCKED and not reason:
        return {
            "ok": False,
            "error": "reason is required when blocking a goal",
        }
    if len(reason) > MAX_BLOCKED_REASON_CHARS:
        return {
            "ok": False,
            "error": (
                "blocked reason exceeds "
                f"{MAX_BLOCKED_REASON_CHARS} characters"
            ),
        }
    goal = current_goal()
    if not goal or str(goal.get("id")) != goal_id:
        _log("goal_stale_update", goal)
        return {
            "ok": False,
            "error": (
                "stale goal id; goal was replaced"
            ),
        }
    if goal.get("status") != GOAL_ACTIVE:
        return {
            "ok": False,
            "error": (
                f"goal is {goal.get('status')}; only an active goal "
                "can be updated by the agent"
            ),
        }
    if objective:
        if len(objective) > MAX_GOAL_CHARS:
            return {
                "ok": False,
                "error": (
                    f"goal objective exceeds {MAX_GOAL_CHARS} characters"
                ),
            }
        goal["objective"] = objective
        _touch(goal)
        save_state()
        _log("goal_updated", goal)
    if not status:
        return {"ok": True, "goal": get_goal_tool_result()["goal"]}
    if status == GOAL_COMPLETE:
        complete_goal(goal_id)
        return {
            "ok": True,
            "goal": get_goal_tool_result()["goal"],
        }
    block_goal(goal_id, reason)
    return {
        "ok": True,
        "goal": get_goal_tool_result()["goal"],
    }


async def set_goal_tool(
    args: dict[str, Any],
    *,
    chat_id: ConversationId | None = None,
) -> dict[str, Any]:
    if thread_active():
        return {
            "ok": False,
            "error": "persistent goals are unavailable inside a thread",
        }
    if not session.goal_creation_authorized:
        return {
            "ok": False,
            "error": (
                "goal creation was not authorized "
                "by the current user turn"
            ),
        }
    existing = current_goal()
    if (
        existing
        and existing.get("status")
        in {
            GOAL_ACTIVE,
            GOAL_PAUSED,
            GOAL_BLOCKED,
        }
    ):
        return {
            "ok": False,
            "error": "an unfinished goal already exists",
        }
    objective = str(args.get("objective") or "")
    try:
        goal = replace_goal(objective)
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
        }
    if chat_id is not None:
        await ensure_goal_pin(chat_id)
        await chat_runtime.send(
            chat_id,
            f"Goal started: {goal['objective']}",
        )
    return {
        "ok": True,
        "goal": get_goal_tool_result()["goal"],
    }
