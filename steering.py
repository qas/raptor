"""Queued steering controls."""
import asyncio
import re
import time

from chat_provider import IncomingAction
from chat_runtime import bound_delivery_context, get_chat_provider, send
import session
from presentation import clear_steering_indicator
from controller import start_root_session
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


def remove_persisted_steer(text: str) -> None:
    pending = session.state.get("pending_inputs")
    if not isinstance(pending, list):
        return
    try:
        pending.remove(text)
    except ValueError:
        return
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
        remove_persisted_steer(str(entry.get("text") or ""))
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
    active = session.active_task
    if (
        active
        and not active.done()
    ):
        active.cancel()
        try:
            await active
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log_exception("steering", "force_wait_error", exc)
    session.pending_steers.pop(
        steer_id,
        None,
    )
    await clear_steering_indicator(
        entry["chat_id"],
        entry.get("message_id"),
        steer_id,
    )
    session.active_since = (
        time.monotonic()
    )
    start_root_session(
        entry["chat_id"],
        str(entry["text"]),
        delivery_context=entry.get("delivery_context"),
    )
    return True
