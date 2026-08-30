"""Tool-approval flow."""
import asyncio
import os
import re
from typing import Any

from chat_provider import ActionButton, ConversationId, IncomingAction
from chat_runtime import get_chat_provider
from presentation import (
    clear_pinned_status,
    create_pinned_status,
    delete_pinned_status,
    show_pinned_status,
)
from session import APPROVAL_TOOLS, pending_approvals, state
from observability import log_agent_activity, log_exception
from tools import execute_tool
from goals import suspend_goal_pin, sync_goal_pin
from tool_activity import tool_preview


def approval_enabled() -> bool:
    return state.get("approval_mode") == "on"


def approval_required(
    call: dict[str, Any],
) -> bool:
    return (
        approval_enabled()
        and call.get("name") in APPROVAL_TOOLS
    )


async def finalize_approval_message(
    entry: dict[str, Any],
    status: str,
) -> None:
    if entry.get("ui_finalized"):
        return

    entry["ui_finalized"] = True

    chat_id = entry.get("chat_id")
    presentation_chat_id = entry.get("presentation_chat_id") or chat_id
    approval_id = str(entry.get("id") or "")

    log_agent_activity(
        f"approval UI finalized: {status}"
    )

    if chat_id is None:
        return

    try:
        if presentation_chat_id != chat_id:
            message_id = entry.get("message_id")
            if message_id is not None:
                await delete_pinned_status(
                    presentation_chat_id,
                    message_id,
                )
        else:
            await sync_goal_pin(
                chat_id,
                released_owner="approval:" + approval_id,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception("approval", "status_cleanup_error", exc)


async def request_tool_approval(
    chat_id: ConversationId,
    call: dict[str, Any],
    *,
    presentation_chat_id: ConversationId | None = None,
) -> str:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = (
        loop.create_future()
    )

    approval_id = os.urandom(
        6
    ).hex()
    preview = tool_preview(
        call
    )

    entry: dict[str, Any] = {
        "id": approval_id,
        "chat_id": chat_id,
        "presentation_chat_id": presentation_chat_id or chat_id,
        "message_id": None,
        "call": call,
        "preview": preview,
        "future": future,
        "ui_finalized": False,
    }
    pending_approvals[approval_id] = entry

    try:
        controls = ((
            ActionButton(
                "✅ Approve",
                f"approval:{approval_id}:approve",
            ),
            ActionButton(
                "❌ Deny",
                f"approval:{approval_id}:deny",
            ),
        ),)
        if (
            presentation_chat_id is not None
            and presentation_chat_id != chat_id
        ):
            message_id = await create_pinned_status(
                presentation_chat_id,
                "⚠️ Approval required\n\n" + preview,
                controls,
            )
        else:
            message_id = await show_pinned_status(
                chat_id,
                "approval:" + approval_id,
                "⚠️ Approval required\n\n" + preview,
                controls=controls,
            )
        entry["message_id"] = message_id
        if presentation_chat_id is None or presentation_chat_id == chat_id:
            await suspend_goal_pin(chat_id)
    except asyncio.CancelledError:
        pending_approvals.pop(approval_id, None)
        if not future.done():
            future.cancel()
        await finalize_approval_message(entry, "cancelled")
        raise
    except Exception:
        pending_approvals.pop(approval_id, None)
        await finalize_approval_message(entry, "failed")
        raise

    try:
        return await future

    except asyncio.CancelledError:
        if not future.done():
            future.cancel()

        await finalize_approval_message(
            entry,
            "⏹ Cancelled",
        )
        raise

    finally:
        pending_approvals.pop(
            approval_id,
            None,
        )


async def execute_tool_with_approval(
    chat_id: ConversationId,
    call: dict[str, Any],
    *,
    execution_context: dict[str, Any]
    | None = None,
    presentation_chat_id: ConversationId | None = None,
) -> dict[str, Any]:
    if not approval_required(
        call
    ):
        return await execute_tool(
            call,
            chat_id=chat_id,
            execution_context=(
                execution_context
            ),
        )

    log_agent_activity(
        "waiting for approval"
    )
    decision = await request_tool_approval(
        chat_id,
        call,
        presentation_chat_id=presentation_chat_id,
    )

    if decision == "approve":
        log_agent_activity(
            "tool approved"
        )
        return await execute_tool(
            call,
            chat_id=chat_id,
            execution_context=(
                execution_context
            ),
        )

    if decision == "steer":
        log_agent_activity(
            "approval superseded by steering"
        )
        return {
            "ok": False,
            "error": (
                "Tool execution cancelled because a steering message "
                "superseded the pending approval."
            ),
            "approval": "superseded",
        }

    log_agent_activity(
        "tool denied"
    )
    return {
        "ok": False,
        "error": "Tool execution denied by user.",
        "approval": "denied",
    }


async def supersede_pending_approvals(
    chat_id: ConversationId,
) -> int:
    superseded = 0

    for entry in list(
        pending_approvals.values()
    ):
        if entry.get("chat_id") != chat_id:
            continue

        future = entry.get(
            "future"
        )

        if not isinstance(
            future,
            asyncio.Future,
        ) or future.done():
            continue

        # The steer must already be queued before this future is resolved so
        # the agent cannot wake and perform another model call without it.
        future.set_result(
            "steer"
        )
        superseded += 1

        await finalize_approval_message(
            entry,
            "↪️ Superseded by steering",
        )

    return superseded


async def _answer_action(
    action_id: str,
    text: str = "",
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
        log_exception("approval", "action_answer_error", exc)


async def handle_approval_action(action: IncomingAction) -> bool:
    provider = get_chat_provider()

    if action.sender_id != provider.authorized_user_id:
        await _answer_action(
            action.action_id,
            "Not authorized.",
            alert=True,
        )
        return True

    data = action.data

    match = re.fullmatch(
        r"approval:([0-9a-f]+):(approve|deny)",
        data,
    )

    if not match:
        return False

    approval_id, decision = (
        match.groups()
    )
    entry = pending_approvals.get(
        approval_id
    )

    callback_chat_id = action.conversation_id

    if (
        not entry
        or entry.get("chat_id")
        != callback_chat_id
    ):
        await _answer_action(
            action.action_id,
            "Approval is no longer pending.",
        )

        presentation_chat_id = (
            action.presentation_conversation_id or callback_chat_id
        )
        if presentation_chat_id is not None:
            if presentation_chat_id != callback_chat_id:
                if action.message_id is not None:
                    await delete_pinned_status(
                        presentation_chat_id,
                        action.message_id,
                    )
            else:
                await clear_pinned_status(
                    presentation_chat_id,
                    owner="approval:" + approval_id,
                )
        return True

    future = entry.get(
        "future"
    )

    if not isinstance(
        future,
        asyncio.Future,
    ) or future.done():
        await _answer_action(action.action_id, "Already handled.")
        return True

    await _answer_action(
        action.action_id,
        "Approved" if decision == "approve" else "Denied",
    )

    future.set_result(
        decision
    )

    await finalize_approval_message(
        entry,
        (
            "✅ Approved"
            if decision == "approve"
            else "❌ Denied"
        ),
    )

    return True
