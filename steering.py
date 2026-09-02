"""Queued steering controls."""
import asyncio
import re

from raptor.chat.chat_provider import IncomingAction
from raptor.chat.chat_runtime import bound_delivery_context, get_chat_provider, send
from raptor.state import session
from raptor.chat.presentation import clear_steering_indicator
from controller import (
    ensure_root_session,
    interrupt_root_turn,
    start_root_session,
)
from observability import log_event, log_exception


async def answer_action(
    action_id: str,
    text: str,
    *,
    alert: bool = False,
) -> None:
    if not action_id:
        return
    try:
        await get_chat_provider().answer_action(
            action_id,
            text,
            alert=alert,
        )
    except Exception as exc:
        log_exception("steering", "action_answer_error", exc)


async def delete_steered_message(entry: dict) -> None:
    """Best-effort removal of the user's cancelled steering message."""
    message_id = entry.get("source_message_id")
    if message_id is None:
        return
    try:
        await get_chat_provider().delete_message(
            entry["chat_id"],
            message_id,
        )
    except Exception as exc:
        log_exception("steering", "message_delete_error", exc)


async def _cancel_steers(*, preserve_forced: bool) -> int:
    preserved_statuses = {"forcing", "force_pending"}
    entries = list(session.pending_steers.values())
    cancelled = [
        entry
        for entry in entries
        if not (
            preserve_forced
            and entry.get("status") in preserved_statuses
        )
    ]
    for entry in cancelled:
        entry["status"] = "cancelled"
        session.pending_steers.pop(str(entry.get("id") or ""), None)
        await clear_steering_indicator(
            entry["chat_id"],
            entry.get("message_id"),
            str(entry.get("id") or ""),
        )
    queued: list[dict] = []
    while True:
        try:
            entry = session.steer_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        session.steer_queue.task_done()
        if (
            preserve_forced
            and entry.get("status") in preserved_statuses
        ):
            queued.append(entry)
    for entry in queued:
        await session.steer_queue.put(entry)
    session.state["pending_inputs"] = [
        {
            "id": str(entry.get("id") or ""),
            "text": str(entry.get("text") or ""),
        }
        for entry in session.pending_steers.values()
        if entry.get("id") and str(entry.get("text") or "")
    ]
    session.save_state()
    return len(cancelled)


async def cancel_pending_steers() -> int:
    """Discard every queued steer and its presentation state."""
    return await _cancel_steers(preserve_forced=False)


async def cancel_unforced_steers() -> int:
    """Discard queued steers while retaining an explicit forced steer."""
    return await _cancel_steers(preserve_forced=True)


def remove_persisted_steer(steer_id: str) -> None:
    pending = session.state.get("pending_inputs")
    if not isinstance(pending, list):
        return
    session.state["pending_inputs"] = [
        item
        for item in pending
        if not isinstance(item, dict) or str(item.get("id") or "") != steer_id
    ]
    session.save_state()


async def handle_steering_action(
    event: IncomingAction,
) -> bool:
    data = event.data
    match = re.fullmatch(
        r"steer:([0-9a-f]+):(apply|cancel)",
        data,
    )
    if not match:
        return False
    if event.sender_id != get_chat_provider().authorized_user_id:
        await answer_action(
            event.action_id,
            "Not authorized.",
            alert=True,
        )
        return True
    steer_id, action = (
        match.groups()
    )
    entry = session.pending_steers.get(
        steer_id
    )
    callback_chat_id = event.conversation_id
    if (
        not entry
        or entry.get("chat_id")
        != callback_chat_id
        or entry.get("status")
        != "queued"
    ):
        await answer_action(
            event.action_id,
            "Steering is no longer pending.",
        )
        return True
    if action == "cancel":
        entry["status"] = "cancelled"
        session.pending_steers.pop(
            steer_id,
            None,
        )
        await answer_action(
            event.action_id,
            "Steering cancelled.",
        )
        await clear_steering_indicator(
            entry["chat_id"],
            entry.get("message_id"),
            steer_id,
        )
        await delete_steered_message(entry)
        remove_persisted_steer(steer_id)
        delivery_context = entry.get("delivery_context")
        if delivery_context is not None:
            with bound_delivery_context(
                entry["chat_id"],
                delivery_context,
            ):
                await send(entry["chat_id"], "Steering cancelled.")
        log_event(
            "agent",
            "steering_cancelled",
            {
                "steer_id": steer_id,
            },
        )
        return True
    entry["status"] = "forcing"
    await answer_action(
        event.action_id,
        "Interrupting and applying now.",
    )
    log_event(
        "agent",
        "steering_forced",
        {
            "steer_id": steer_id,
        },
    )
    interrupted = await interrupt_root_turn()
    if interrupted.error is not None:
        log_exception("steering", "force_wait_error", interrupted.error)
    if interrupted.completed:
        session.persist_steer_handoff(entry)
        await clear_steering_indicator(
            entry["chat_id"],
            entry.get("message_id"),
            steer_id,
        )
        start_root_session(
            entry["chat_id"],
            str(entry["text"]),
            input_recorded=True,
            delivery_context=entry.get("delivery_context"),
        )
    else:
        # Keep ownership singular while cancellation drains. The root's
        # lost-wakeup guard applies this steer after the old owner exits.
        entry["status"] = "force_pending"
        ensure_root_session(entry["chat_id"], None)
    return True
