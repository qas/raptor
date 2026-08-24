"""Projection of temporary-thread state onto the shared status surface."""

from chat_provider import ActionButton, ConversationId
import presentation
import session
from thread_state import current_thread, thread_owner


def _approval_status_active(
    conversation_id: ConversationId,
    *,
    ignore_owner: str | None = None,
) -> bool:
    runtime = session.current_runtime()
    owner = runtime.pinned_status_owner
    if (
        runtime.pinned_status_conversation_id == conversation_id
        and owner != ignore_owner
        and isinstance(owner, str)
        and owner.startswith("approval:")
    ):
        return True
    return any(
        entry.get("chat_id") == conversation_id
        and not entry.get("ui_finalized")
        and entry.get("message_id") is not None
        for entry in session.pending_approvals.values()
    )


async def ensure_thread_status(
    conversation_id: ConversationId,
    *,
    replace_owner: str | None = None,
) -> None:
    thread = current_thread()
    owner = thread_owner(thread)
    if not thread or not owner:
        return
    if _approval_status_active(
        conversation_id,
        ignore_owner=replace_owner,
    ):
        return
    runtime = session.current_runtime()
    if (
        runtime.pinned_status_conversation_id == conversation_id
        and runtime.pinned_status_owner == owner
        and runtime.pinned_status_message_id is not None
    ):
        return
    await presentation.show_pinned_status(
        conversation_id,
        owner,
        (
            "Thread active\n"
            "Clear discards this branch; Merge adds it to the main conversation."
        ),
        controls=((
            ActionButton("✖ Clear", f"{owner}:clear"),
            ActionButton("↩ Merge", f"{owner}:merge"),
        ),),
    )
    current = current_thread()
    if not current or str(current.get("id")) != str(thread.get("id")):
        await presentation.clear_pinned_status(
            conversation_id,
            owner=owner,
        )
        return
    runtime.goal_pin_message_id = None
    runtime.goal_pin_goal_id = None
