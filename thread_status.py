"""Projection of temporary-thread state onto the shared status surface."""

from chat_provider import ActionButton, ConversationId
import presentation
from raptor.state import session
from thread_state import current_thread, thread_owner


async def ensure_thread_status(
    conversation_id: ConversationId,
) -> None:
    thread = current_thread()
    owner = thread_owner(thread)
    if not thread or not owner:
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
