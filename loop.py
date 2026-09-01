"""Provider-neutral incoming chat-event dispatcher."""
import secrets

from approval import handle_approval_action, supersede_pending_approvals
from chat_provider import (
    ChatEvent,
    ChatProvider,
    IncomingAction,
    IncomingMessage,
)
from chat_runtime import capture_delivery_context, get_chat_provider
from chat_store import append_meta
from commands import command
from config import MAX_PENDING_STEERS
from controller import start_root_session
from presentation import clear_steering_indicator, steering_indicator
from session import StateCapacityError, state, steer_queue
from steering import handle_steering_action
from threads import handle_thread_action
import session
from observability import log_agent_activity, log_event, log_exception
from turn_runtime import turns

COMMANDS: tuple[tuple[str, str], ...] = (
    ("new", "New session"),
    ("chats", "List or search sessions"),
    ("resume", "Resume a prior session"),
    ("ask", "Ask without session context"),
    ("thread", "Temporary conversation branch"),
    ("status", "Show status"),
    ("stop", "Abort current run"),
    ("compact", "Compact context"),
    ("truncate", "Remove recent user turns"),
    ("model", "List/switch model"),
    ("models", "List models from provider"),
    ("approval", "Toggle tool approval"),
    ("todos", "Show todo list"),
    ("subagents", "Show subagent status"),
    ("console", "Run a managed shell command"),
    ("shutdown", "Shut down Raptor"),
    ("restart", "Restart Raptor"),
    ("goal", "Show or set persistent goal"),
    ("help", "Show commands"),
)


def accepts_event(event: ChatEvent, provider: ChatProvider) -> bool:
    """Return whether an event may enter chat-scoped runtime state."""
    return bool(
        event.conversation_id is not None
        and event.sender_id == provider.authorized_user_id
        and event.interactive
    )


async def handle_event(event: ChatEvent) -> None:
    """Dispatch a normalized event without provider-specific payloads."""
    provider = get_chat_provider()

    if not accepts_event(event, provider):
        return

    if isinstance(event, IncomingAction):
        if await handle_thread_action(event):
            return
        if await handle_steering_action(event):
            return
        if await handle_approval_action(event):
            return
        try:
            await provider.answer_action(event.action_id)
        except Exception as exc:
            log_exception(provider.name, "action_answer_error", exc)
        return

    if not isinstance(event, IncomingMessage):
        return
    conversation_id = event.conversation_id
    text = event.text.strip()
    if not text:
        return

    command_name = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
    received_data = {
        "conversation_id": conversation_id,
        "message_id": event.message_id,
        "command": command_name if command_name.startswith("/") else None,
        "text_chars": len(text),
    }
    log_event(provider.name, "received", received_data)

    if text.startswith("/"):
        command_result = await command(conversation_id, text)
        if command_result is True:
            return
        if isinstance(command_result, str):
            text = command_result

    if turns.is_running():
        if await provider.reject_busy_message(conversation_id):
            log_agent_activity("rejected concurrent request")
            return
        queued_steers = sum(
            entry.get("status") == "queued"
            for entry in session.pending_steers.values()
        )
        if queued_steers >= MAX_PENDING_STEERS:
            await provider.send_text(
                conversation_id,
                f"Steering queue is full ({MAX_PENDING_STEERS}).",
            )
            log_agent_activity("rejected full steering queue")
            return
        steer_id = secrets.token_hex(4)
        indicator_id = await steering_indicator(
            conversation_id,
            steer_id,
        )
        entry = {
            "id": steer_id,
            "chat_id": conversation_id,
            "text": text,
            "source_message_id": event.message_id,
            "message_id": indicator_id,
            "status": "queued",
            "delivery_context": capture_delivery_context(conversation_id),
        }
        session_id = state.get("current_session_id")
        if session_id:
            try:
                session.queue_pending_steer(steer_id, text)
            except StateCapacityError:
                await clear_steering_indicator(
                    conversation_id,
                    indicator_id,
                    steer_id,
                )
                await provider.send_text(
                    conversation_id,
                    "Steering input is too large to queue safely.",
                )
                log_agent_activity("rejected oversized steering input")
                return
            append_meta(
                str(session_id),
                "steer_queued",
                {"steer_id": steer_id},
            )
        session.pending_steers[steer_id] = entry
        await steer_queue.put(entry)
        await supersede_pending_approvals(conversation_id)
        await provider.acknowledge_queued_message(conversation_id)
        log_agent_activity("queued steering input")
        return

    start_root_session(
        conversation_id,
        text,
        delivery_context=capture_delivery_context(conversation_id),
        source_message_id=event.message_id,
    )
