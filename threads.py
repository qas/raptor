"""Temporary conversation branches with explicit clear/merge transitions."""
import copy
import re
import secrets
import time
from typing import Any

from chat_provider import ConversationId, IncomingAction
from chat_runtime import get_chat_provider
from chat_store import (
    active_item_events,
    append_event,
    append_item,
    append_meta,
    create_session,
    end_session,
    iter_events,
    session_exists,
)
from context import build_active_context
from controller import ensure_root_session
from goals import goal_is_active, sync_goal_pin
import session
from observability import log_event, log_exception
from thread_state import current_thread, thread_active, thread_owner
from thread_status import ensure_thread_status
from controller import session_transition_busy


def thread_busy() -> bool:
    return session_transition_busy()


async def start_thread(conversation_id: ConversationId) -> dict[str, Any]:
    existing = current_thread()
    if existing:
        return {"ok": False, "error": "A thread is already active."}
    if thread_busy():
        return {"ok": False, "error": "Busy. Use /stop all first."}
    parent_id = str(session.state.get("current_session_id") or "")
    if not session_exists(parent_id):
        return {"ok": False, "error": "No valid main session."}
    branch_id = create_session(
        kind="thread",
        chat_key=session.current_runtime().key,
        parent_session_id=parent_id,
    )
    seed = build_active_context(parent_id)
    for item in seed:
        append_item(
            branch_id,
            copy.deepcopy(item),
            source="thread_seed",
        )
    thread = {
        "id": secrets.token_hex(4),
        "conversation_id": conversation_id,
        "parent_session_id": parent_id,
        "session_id": branch_id,
        "started_at": time.time(),
        "seed_items": len(seed),
        "parent_interrupted_subagents": copy.deepcopy(
            session.state.get("interrupted_subagents") or []
        ),
    }
    append_meta(
        branch_id,
        "thread_fork",
        {
            "thread_id": thread["id"],
            "parent_session_id": parent_id,
            "seed_items": len(seed),
        },
    )
    session.state["thread"] = thread
    session.state["current_session_id"] = branch_id
    session.state["interrupted_subagents"] = []
    session.save_state()
    try:
        await ensure_thread_status(conversation_id)
    except Exception as exc:
        session.state["thread"] = None
        session.state["current_session_id"] = parent_id
        session.state["interrupted_subagents"] = thread[
            "parent_interrupted_subagents"
        ]
        session.save_state()
        end_session(branch_id, reason="thread_start_failed")
        return {
            "ok": False,
            "error": f"Could not show thread status: {type(exc).__name__}: {exc}",
        }
    append_meta(
        parent_id,
        "thread_started",
        {"thread_id": thread["id"], "session_id": branch_id},
    )
    return {"ok": True, "thread": copy.deepcopy(thread)}


async def finish_thread(
    conversation_id: ConversationId,
    *,
    merge: bool,
) -> dict[str, Any]:
    thread = current_thread()
    owner = thread_owner(thread)
    if not thread or not owner:
        return {"ok": False, "error": "No thread is active."}
    if thread_busy():
        return {"ok": False, "error": "Busy. Use /stop all first."}
    parent_id = str(thread.get("parent_session_id") or "")
    branch_id = str(thread.get("session_id") or "")
    if not session_exists(parent_id) or not session_exists(branch_id):
        return {"ok": False, "error": "Thread session state is invalid."}
    merged = 0
    if merge:
        existing_origins = {
            (
                str(origin.get("session_id") or ""),
                int(origin.get("seq") or 0),
            )
            for event in iter_events(parent_id)
            if event.get("source") == "thread_merge"
            and isinstance((origin := event.get("origin")), dict)
            and str(origin.get("session_id") or "") == branch_id
        }
        for event in active_item_events(branch_id):
            if event.get("source") == "thread_seed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            origin = (branch_id, int(event.get("seq") or 0))
            if origin in existing_origins:
                continue
            append_event(
                parent_id,
                {
                    "type": "item",
                    "source": "thread_merge",
                    "origin": {
                        "session_id": origin[0],
                        "seq": origin[1],
                    },
                    "item": copy.deepcopy(item),
                },
            )
            existing_origins.add(origin)
            merged += 1
    end_session(
        branch_id,
        reason="thread_merged" if merge else "thread_cleared",
    )
    append_meta(
        parent_id,
        "thread_merged" if merge else "thread_cleared",
        {
            "thread_id": thread.get("id"),
            "session_id": branch_id,
            "merged_items": merged,
        },
    )
    session.state["thread"] = None
    session.state["current_session_id"] = parent_id
    session.state["interrupted_subagents"] = copy.deepcopy(
        thread.get("parent_interrupted_subagents") or []
    )
    session.save_state()
    try:
        await sync_goal_pin(conversation_id, released_owner=owner)
    except Exception as exc:
        log_event(
            "thread",
            "thread_pin_restore_error",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
    return {"ok": True, "merged_items": merged}


def resume_main_goal(conversation_id: ConversationId) -> None:
    if goal_is_active() and not thread_active():
        ensure_root_session(conversation_id, None)


async def _answer_action(
    action_id: str,
    text: str,
    *,
    alert: bool = False,
) -> None:
    try:
        await get_chat_provider().answer_action(
            action_id,
            text,
            alert=alert,
        )
    except Exception as exc:
        log_exception("thread", "action_answer_error", exc)


async def handle_thread_action(action: IncomingAction) -> bool:
    match = re.fullmatch(
        r"thread:([0-9a-f]+):(clear|merge)",
        action.data,
    )
    if not match:
        return False
    provider = get_chat_provider()
    if action.sender_id != provider.authorized_user_id:
        await _answer_action(
            action.action_id,
            "Not authorized.",
            alert=True,
        )
        return True
    thread_id, decision = match.groups()
    thread = current_thread()
    if (
        not thread
        or str(thread.get("id")) != thread_id
        or action.conversation_id != thread.get("conversation_id")
    ):
        await _answer_action(
            action.action_id,
            "Thread is no longer active.",
        )
        return True
    result = await finish_thread(
        action.conversation_id,
        merge=decision == "merge",
    )
    if not result["ok"]:
        await _answer_action(
            action.action_id,
            str(result["error"]),
            alert=True,
        )
        return True
    await _answer_action(
        action.action_id,
        (
            f"Thread merged ({result['merged_items']} items)."
            if decision == "merge"
            else "Thread cleared."
        ),
    )
    resume_main_goal(action.conversation_id)
    return True
