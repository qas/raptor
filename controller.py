"""Root session controller — sole owner of root turn scheduling."""
import asyncio
from collections.abc import Callable
from typing import Any, Literal

from raptor.state import session
from agent import (
    RetryableTurnFailure,
    agent_turn,
    compact_context,
    flush_pending_delivery,
)
from raptor.chat.chat_provider import ConversationId
from raptor.chat.chat_runtime import bound_delivery_context, send
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
from observability import log_event
from raptor.chat.presentation import clear_steering_indicator
from runtime_events import RuntimeEvent, RuntimeEventKind
from raptor.state.session import save_state
from thread_state import thread_active
from turn_runtime import InterruptResult, TurnKind, TurnSnapshot, turns

WorkSource = Literal["user", "runtime", "goal", "internal"]
WorkEntry = RuntimeEvent | dict[str, Any]


def session_transition_busy() -> bool:
    """Return whether changing the active transcript could orphan work."""
    from raptor.shell.shell_sessions import (
        pending_shell_completions,
        running_shell_sessions,
    )
    from subagents import pending_subagent_completions

    return bool(
        session.current_runtime().state.get("session_transition")
        or turns.is_running()
        or session.subagent_tasks
        or running_shell_sessions()
        or session.pending_approvals
        or session.pending_steers
        or not session.runtime_event_queue.empty()
        or pending_subagent_completions()
        or pending_shell_completions()
    )


async def requeue_deferred_completions() -> int:
    """Retry background completion delivery on explicit user activity."""
    from raptor.shell.shell_sessions import requeue_deferred_shell_completions
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
        if entry.get("status") not in {"queued", "force_pending"}:
            if entry.get("status") != "forcing":
                await clear_steering_indicator(
                    entry["chat_id"],
                    entry.get("message_id"),
                    str(entry.get("id") or ""),
                )
            continue
        previous_status = str(entry["status"])
        steer_id = str(entry["id"])
        entry["status"] = "applied"
        session.pending_steers.pop(steer_id, None)
        try:
            await clear_steering_indicator(
                entry["chat_id"],
                entry.get("message_id"),
                steer_id,
            )
        except asyncio.CancelledError:
            entry["status"] = previous_status
            session.pending_steers[steer_id] = entry
            session.steer_queue.put_nowait(entry)
            raise
        session.persist_steer_handoff(entry)
        return entry


def _dequeue_runtime_event() -> RuntimeEvent | None:
    while True:
        try:
            event = session.runtime_event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        session.runtime_event_queue.task_done()
        if event.is_active is not None and not event.is_active():
            if not event.done.done():
                event.done.set_result(False)
            continue
        return event


def _pending_controller_work() -> bool:
    if (
        goal_is_active()
        and not thread_active()
        and not session.subagent_tasks
    ):
        return True
    if session.runtime_event_queue.qsize() > 0:
        return True
    for entry in session.pending_steers.values():
        if entry.get("status") in {"queued", "force_pending"}:
            return True
    return False


def discard_runtime_events() -> int:
    """Discard queued background notifications and release their producers."""
    discarded = 0
    while True:
        try:
            event = session.runtime_event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        session.runtime_event_queue.task_done()
        if not event.done.done():
            event.done.set_result(False)
        discarded += 1
    return discarded


async def _select_next_work(
    captured_goal_id: str | None,
) -> tuple[
    str | None,
    WorkSource | None,
    WorkEntry | None,
]:
    steer = await _dequeue_steer()
    if steer is not None:
        return str(steer["text"]), "user", steer
    runtime_event = _dequeue_runtime_event()
    if runtime_event is not None:
        return (
            runtime_event.prompt(),
            "runtime",
            runtime_event,
        )
    if (
        goal_is_active()
        and not thread_active()
        and not session.subagent_tasks
    ):
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


def _finish_runtime_event(
    event: RuntimeEvent | None,
    delivered: bool,
) -> None:
    if event is None:
        return
    if not event.done.done():
        event.done.set_result(delivered)


def _mark_goal_continuation(
    source: WorkSource | None,
) -> None:
    if source == "goal":
        turns.set_goal_id(current_goal_id())
    else:
        turns.set_goal_id(None)


async def run_root_session(
    chat_id: ConversationId,
    initial_input: str | None,
    *,
    internal: bool = False,
    input_recorded: bool = False,
    delivery_context: Any | None = None,
    source_message_id: int | str | None = None,
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
    work_entry: RuntimeEvent | dict[str, Any] | None = None
    captured_goal_id = (
        current_goal_id()
        if goal_is_active() and not thread_active()
        else None
    )
    delivery_blocked = False
    try:
        if not await flush_pending_delivery(chat_id):
            delivery_blocked = True
            return
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
            if isinstance(work_entry, RuntimeEvent):
                chat_id = work_entry.conversation_id
            elif isinstance(work_entry, dict):
                chat_id = work_entry.get("chat_id", chat_id)
                delivery_context = work_entry.get("delivery_context")
                input_recorded = True
            _mark_goal_continuation(next_source)
            if next_source == "goal":
                await ensure_goal_pin(chat_id)
            allow_goal_creation = (
                next_source == "user" and not thread_active()
            )
            try:
                with bound_delivery_context(chat_id, delivery_context):
                    turn_options = {
                        "internal": next_source != "user",
                        "source": next_source or "user",
                        "allow_goal_creation": allow_goal_creation,
                    }
                    if input_recorded:
                        turn_options["input_recorded"] = True
                    if next_source == "user" and source_message_id is not None:
                        turn_options["source_message_id"] = source_message_id
                    delivered = await agent_turn(
                        chat_id,
                        next_input,
                        **turn_options,
                    )
            except asyncio.CancelledError:
                _finish_runtime_event(
                    work_entry
                    if isinstance(work_entry, RuntimeEvent)
                    else None,
                    False,
                )
                raise
            _finish_runtime_event(
                work_entry
                if isinstance(work_entry, RuntimeEvent)
                else None,
                delivered is True,
            )
            work_entry = None
            delivery_context = None
            source_message_id = None
            input_recorded = False
            if session.state.get("pending_delivery") is not None:
                if isinstance(delivered, RetryableTurnFailure):
                    pause_goal()
                break
            if isinstance(delivered, RetryableTurnFailure) or delivered is None:
                reason = (
                    delivered.reason
                    if isinstance(delivered, RetryableTurnFailure)
                    else "a temporary agent failure"
                )
                if goal_is_active():
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
                and next_source == "goal"
                and goal_is_active()
            ):
                goal_id = current_goal_id()
                if goal_id:
                    block_goal(
                        goal_id,
                        "unrecoverable agent turn failure",
                    )
            await _announce_goal_terminal(chat_id)
            await sync_goal_pin(chat_id)
            if (
                captured_goal_id
                and goal_is_active()
                and current_goal_id()
                != captured_goal_id
            ):
                break
            (
                next_input,
                next_source,
                work_entry,
            ) = await _select_next_work(
                captured_goal_id
            )
            if next_input is None:
                break
    except Exception:
        _finish_runtime_event(
            work_entry if isinstance(work_entry, RuntimeEvent) else None,
            False,
        )
        raise
    finally:
        turns.set_goal_id(None)
        snapshot = turns.snapshot
        if turns.finish(asyncio.current_task()):
            marker = session.state.get("active_root_turn")
            if (
                isinstance(marker, dict)
                and snapshot is not None
                and marker.get("id") == snapshot.id
            ):
                session.state["active_root_turn"] = None
                save_state()
            # Lost-wakeup guard: work may have arrived after the idle
            # decision and before turn ownership was released.
            if not delivery_blocked and _pending_controller_work():
                start_root_session(chat_id, None)


def start_root_session(
    chat_id: ConversationId,
    text: str | None,
    *,
    internal: bool = False,
    input_recorded: bool = False,
    delivery_context: Any | None = None,
    source_message_id: int | str | None = None,
) -> asyncio.Task[None]:
    def persist_turn(snapshot: TurnSnapshot) -> None:
        session.set_active_root_turn(
            {
                "id": snapshot.id,
                "session_id": session.state.get("current_session_id"),
            }
        )

    return turns.start(
        run_root_session(
            chat_id,
            text,
            internal=internal,
            input_recorded=input_recorded,
            delivery_context=delivery_context,
            source_message_id=source_message_id,
        ),
        kind=TurnKind.REGULAR,
        before_start=persist_turn,
    )


def ensure_root_session(
    chat_id: ConversationId,
    text: str | None = None,
    *,
    internal: bool = False,
    input_recorded: bool = False,
    delivery_context: Any | None = None,
    source_message_id: int | str | None = None,
) -> asyncio.Task | None:
    if turns.is_running():
        return turns.task
    return start_root_session(
        chat_id,
        text,
        internal=internal,
        input_recorded=input_recorded,
        delivery_context=delivery_context,
        source_message_id=source_message_id,
    )


async def _run_manual_compaction(chat_id: ConversationId) -> None:
    cancelled = False
    try:
        await compact_context(chat_id, reason="manual")
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        current_task = asyncio.current_task()
        if turns.task is current_task:
            if cancelled:
                from steering import cancel_unforced_steers

                await cancel_unforced_steers()
            if (
                turns.finish(current_task)
                and _pending_controller_work()
            ):
                ensure_root_session(chat_id)


def start_manual_compaction(chat_id: ConversationId) -> asyncio.Task[None]:
    """Start manual compaction under the root task-ownership boundary."""
    return turns.start(
        _run_manual_compaction(chat_id),
        kind=TurnKind.MANUAL_COMPACTION,
    )


async def interrupt_root_turn(
    *,
    expected_turn_id: str | None = None,
) -> InterruptResult:
    """Interrupt the active root turn without touching background resources."""
    return await turns.interrupt(expected_turn_id=expected_turn_id)


async def interrupt_active_goal_controller(
    goal_id: str | None,
) -> bool:
    """Cancel the root controller only if it is on this goal's continuation."""
    if not goal_id:
        return False
    if turns.goal_id != str(goal_id):
        return False
    result = await interrupt_root_turn()
    return result.interrupted


def enqueue_runtime_event(
    chat_id: ConversationId,
    kind: RuntimeEventKind,
    text: str,
    *,
    is_active: Callable[[], bool] | None = None,
) -> asyncio.Future[bool]:
    with session.bound_chat(chat_id):
        done: asyncio.Future[bool] = (
            asyncio.get_running_loop().create_future()
        )
        event = RuntimeEvent(
            conversation_id=chat_id,
            kind=kind,
            content=text,
            done=done,
            is_active=is_active,
        )
        session.runtime_event_queue.put_nowait(event)
        ensure_root_session(chat_id, None)
        return done
