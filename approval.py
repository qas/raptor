"""Tool-approval flow."""
import asyncio
import os
import re
from typing import Any

from chat_provider import ActionButton, ConversationId, IncomingAction
from chat_runtime import get_chat_provider
from session import APPROVAL_TOOLS, pending_approvals, state
from observability import log_agent_activity, log_exception
from tools import execute_tool
from tool_activity import ToolActivitySurface


def approval_enabled() -> bool:
    return state.get("approval_mode") == "on"


def approval_required(
    call: dict[str, Any],
) -> bool:
    return (
        approval_enabled()
        and call.get("name") in APPROVAL_TOOLS
    )


async def request_tool_approval(
    chat_id: ConversationId,
    call: dict[str, Any],
    *,
    tool_activity: ToolActivitySurface,
) -> str:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = (
        loop.create_future()
    )

    approval_id = os.urandom(
        6
    ).hex()
    entry: dict[str, Any] = {
        "id": approval_id,
        "chat_id": chat_id,
        "message_id": None,
        "call": call,
        "future": future,
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
        message_id = await tool_activity.approval(call, controls)
        entry["message_id"] = message_id
    except asyncio.CancelledError:
        pending_approvals.pop(approval_id, None)
        if not future.done():
            future.cancel()
        raise
    except Exception:
        pending_approvals.pop(approval_id, None)
        raise

    try:
        return await future

    except asyncio.CancelledError:
        if not future.done():
            future.cancel()

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
    tool_activity: ToolActivitySurface | None = None,
) -> dict[str, Any]:
    owned_activity = False

    async def finish_owned_activity(result: dict[str, Any]) -> None:
        if tool_activity is None or not owned_activity:
            return
        await tool_activity.finished(call, result)
        await tool_activity.clear()

    try:
        if approval_required(call):
            log_agent_activity("waiting for approval")
            if tool_activity is None:
                tool_activity = ToolActivitySurface(
                    presentation_chat_id or chat_id,
                )
                owned_activity = True
            decision = await request_tool_approval(
                chat_id,
                call,
                tool_activity=tool_activity,
            )
            if decision != "approve":
                if decision == "steer":
                    log_agent_activity("approval superseded by steering")
                    result = {
                        "ok": False,
                        "error": (
                            "Tool execution cancelled because a steering "
                            "message superseded the pending approval."
                        ),
                        "approval": "superseded",
                    }
                else:
                    log_agent_activity("tool denied")
                    result = {
                        "ok": False,
                        "error": "Tool execution denied by user.",
                        "approval": "denied",
                    }
                await finish_owned_activity(result)
                return result
            log_agent_activity("tool approved")

        if tool_activity is not None:
            await tool_activity.running(call)
        result = await execute_tool(
            call,
            chat_id=chat_id,
            execution_context=execution_context,
        )
    except asyncio.CancelledError:
        await finish_owned_activity(
            {"ok": False, "status": "interrupted"},
        )
        raise
    except Exception:
        await finish_owned_activity({"ok": False})
        raise
    await finish_owned_activity(result)
    return result


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

    return True
