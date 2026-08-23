"""Root session controller — sole owner of root turn scheduling."""
import asyncio
import time
from typing import Any, Literal

from chat_provider import ConversationId

import session
from agent import RetryableTurnFailure, agent_turn, compact_context
from goals import (
    GOAL_BLOCKED,
    GOAL_COMPLETE,
    block_goal,
    current_goal,
    current_goal_id,
    ensure_goal_pin,
    goal_continuation_input,
    goal_is_active,
    pause_goal,
    sync_goal_pin,
)
from session import save_state
from chat_runtime import bound_delivery_context, send
from presentation import clear_steering_indicator
from observability import log_event
from thread_state import thread_active

WorkSource = Literal["user", "internal", "goal"]


async def requeue_deferred_completions() -> int:
    """Retry background completion delivery on explicit user activity."""
    from shell_sessions import requeue_deferred_shell_completions
    from subagents import requeue_deferred_subagent_completions

    shell_count, subagent_count = await asyncio.gather(
        requeue_deferred_shell_completions(),
        requeue_deferred_subagent_completions(),
    )
    return shell_count + subagent_count


async def _dequeue_steer() -> dict[str, Any] | None:
    while True:
        try:
            entry = session.steer_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        session.steer_queue.task_done()
        if entry.get("status") != "queued":
            if entry.get("status") != "forcing":
                await clear_steering_indicator(
                    entry["chat_id"],
                    entry.get("message_id"),
                    str(entry.get("id") or ""),
                )
            continue
        entry["status"] = "applied"
        session.pending_steers.pop(str(entry["id"]), None)
        await clear_steering_indicator(
            entry["chat_id"],
            entry.get("message_id"),
            str(entry.get("id") or ""),
        )
        return entry


def _dequeue_internal() -> dict[str, Any] | None:
    while True:
        try:
            entry = session.internal_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        session.internal_queue.task_done()
        is_active = entry.get("is_active")
        if callable(is_active) and not is_active():
            done = entry.get("done")
            if isinstance(done, asyncio.Future) and not done.done():
                done.set_result(False)
            continue
        return entry


def _pending_controller_work() -> bool:
    if goal_is_active() and not thread_active():
        return True
    if session.internal_queue.qsize() > 0:
        return True
    if session.steer_queue.qsize() > 0:
        return True
    for entry in session.pending_steers.values():
        if entry.get("status") == "queued":
            return True
    return False


async def _select_next_work(
    captured_goal_id: str | None,
) -> tuple[
    str | None,
    WorkSource | None,
    dict[str, Any] | None,
]:
    steer = await _dequeue_steer()
    if steer is not None:
        return str(steer["text"]), "user", steer
    internal = _dequeue_internal()
    if internal is not None:
        return (
            str(internal["text"]),
            "internal",
            internal,
        )
    if goal_is_active() and not thread_active():
        goal_id = current_goal_id()
        if (
            captured_goal_id
            and goal_id
            and captured_goal_id != goal_id
        ):
            return None, None, None
        log_event(
            "goal",
            "goal_continuation",
            {
                "goal_id": goal_id,
                "status": "active",
            },
        )
        return (
            goal_continuation_input(),
            "goal",
            None,
        )
    return None, None, None


async def _announce_goal_terminal(
    chat_id: ConversationId,
) -> None:
    goal = current_goal()
    if not goal:
        return
    status = goal.get("status")
    if status not in {
        GOAL_COMPLETE,
        GOAL_BLOCKED,
    }:
        return
    if goal.get("notified_status") == status:
        return
    objective = str(goal.get("objective") or "")
    if status == GOAL_COMPLETE:
        await send(
            chat_id,
            f"Goal complete: {objective}",
        )
    else:
        reason = str(
            goal.get("blocked_reason")
            or "blocked"
        )
        await send(
            chat_id,
            f"Goal blocked: {reason}",
        )
    goal["notified_status"] = status
    save_state()


def _finish_internal(
    entry: dict[str, Any] | None,
    delivered: bool,
) -> None:
    if not entry:
        return
    done = entry.get("done")
    if isinstance(done, asyncio.Future) and not done.done():
        done.set_result(delivered)


def _mark_goal_continuation(
    source: WorkSource | None,
) -> None:
    if source == "goal":
        session.active_goal_id = current_goal_id()
    else:
        session.active_goal_id = None


async def run_root_session(
    chat_id: ConversationId,
    initial_input: str | None,
    *,
    internal: bool = False,
    delivery_context: Any | None = None,
) -> None:
    next_input = initial_input
    next_source: WorkSource | None = (
        (
            "internal"
            if internal
            else "user"
        )
        if initial_input is not None
        else None
    )
    work_entry: dict[str, Any] | None = None
    captured_goal_id = (
        current_goal_id()
        if goal_is_active() and not thread_active()
        else None
    )
    try:
        if initial_input is not None and not internal:
            await requeue_deferred_completions()
        await ensure_goal_pin(chat_id)
        while True:
            if next_input is None:
                (
                    next_input,
                    next_source,
                    work_entry,
                ) = await _select_next_work(
                    captured_goal_id
                )
                if next_input is None:
                    break
            if work_entry is not None:
                entry_chat_id = work_entry.get("chat_id")
                if entry_chat_id is not None:
                    chat_id = entry_chat_id
                delivery_context = work_entry.get("delivery_context")
            _mark_goal_continuation(next_source)
            if next_source == "goal":
                await ensure_goal_pin(chat_id)
            allow_goal_creation = (
                next_source == "user" and not thread_active()
            )
            try:
                with bound_delivery_context(chat_id, delivery_context):
                    delivered = await agent_turn(
                        chat_id,
                        next_input,
                        internal=(next_source != "user"),
                        source=next_source or "user",
                        allow_goal_creation=allow_goal_creation,
                    )
            except asyncio.CancelledError:
                _finish_internal(work_entry, False)
                raise
            _finish_internal(
                work_entry,
                delivered is True,
            )
            work_entry = None
            if isinstance(delivered, RetryableTurnFailure) or delivered is None:
                reason = (
                    delivered.reason
                    if isinstance(delivered, RetryableTurnFailure)
                    else "a temporary agent failure"
                )
                if goal_is_active() and not thread_active():
                    _goal, changed = pause_goal()
                    if changed:
                        await send(
                            chat_id,
                            f"Goal paused after {reason}. "
                            "Use /goal resume to continue.",
                        )
                await sync_goal_pin(chat_id)
                break
            if (
                delivered is False
                and goal_is_active()
                and not thread_active()
            ):
                goal_id = current_goal_id()
                if goal_id:
                    block_goal(
                        goal_id,
                        "unrecoverable agent turn failure",
                    )
            await _announce_goal_terminal(chat_id)
            await sync_goal_pin(chat_id)
            (
                next_input,
                next_source,
                work_entry,
            ) = await _select_next_work(
                captured_goal_id
            )
            if next_input is None:
                break
            if (
                captured_goal_id
                and goal_is_active()
                and current_goal_id()
                != captured_goal_id
            ):
                break
    except Exception:
        _finish_internal(work_entry, False)
        raise
    finally:
        session.active_goal_id = None
        if session.active_task is asyncio.current_task():
            session.active_task = None
            session.active_since = None
            # Lost-wakeup guard: work may have arrived after the idle
            # decision and before active_task was cleared.
            if _pending_controller_work():
                start_root_session(chat_id, None)


def start_root_session(
    chat_id: ConversationId,
    text: str | None,
    *,
    internal: bool = False,
    delivery_context: Any | None = None,
) -> asyncio.Task:
    session.active_since = time.monotonic()
    task = asyncio.create_task(
        run_root_session(
            chat_id,
            text,
            internal=internal,
            delivery_context=delivery_context,
        )
    )
    session.active_task = task
    return task


def ensure_root_session(
    chat_id: ConversationId,
    text: str | None = None,
    *,
    internal: bool = False,
    delivery_context: Any | None = None,
) -> asyncio.Task | None:
    active = session.active_task
    if active and not active.done():
        return active
    return start_root_session(
        chat_id,
        text,
        internal=internal,
        delivery_context=delivery_context,
    )


async def _run_manual_compaction(chat_id: ConversationId) -> None:
    cancelled = False
    try:
        await compact_context(chat_id, reason="manual")
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        if session.active_task is asyncio.current_task():
            session.active_task = None
            session.active_since = None
            if cancelled:
                while True:
                    entry = await _dequeue_steer()
                    if entry is None:
                        break
                    entry["status"] = "cancelled"
            elif _pending_controller_work():
                ensure_root_session(chat_id)


def start_manual_compaction(chat_id: ConversationId) -> asyncio.Task[None]:
    """Start manual compaction under the root task-ownership boundary."""
    session.active_since = time.monotonic()
    task = asyncio.create_task(_run_manual_compaction(chat_id))
    session.active_task = task
    return task


def cancel_active_goal_controller(
    goal_id: str | None,
) -> asyncio.Task | None:
    """Cancel the root controller only if it is on this goal's continuation."""
    if not goal_id:
        return None
    if session.active_goal_id != str(goal_id):
        return None
    active = session.active_task
    if not active or active.done():
        return None
    active.cancel()
    return active


async def enqueue_internal_input(
    chat_id: ConversationId,
    text: str,
    *,
    is_active: Any | None = None,
) -> bool:
    done: asyncio.Future[bool] = (
        asyncio.get_running_loop().create_future()
    )
    entry = {
        "chat_id": chat_id,
        "text": text,
        "internal": True,
        "done": done,
        "is_active": is_active,
        "delivery_context": None,
    }
    await session.internal_queue.put(entry)
    ensure_root_session(chat_id, None)
    return bool(await done)
