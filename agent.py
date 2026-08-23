"""Agent turn and context compaction entry points."""
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from chat_provider import ConversationId

from chat_store import append_item, append_meta
from config import (
    COMPACTION_REASONING_EFFORT,
    MAX_TOOL_ROUNDS,
    compaction_generation_budget,
    context_input_budget,
)
from context import (
    build_active_context,
    compact_session,
    ensure_context_under_budget,
    request_with_checkpoint_retry,
    session_context_stats,
)
from engine import run_agent
from goals import (
    combine_instructions,
    goal_instructions,
    todo_store_for_execution,
)
from session import save_state, state, steer_queue
import session
from approval import execute_tool_with_approval
from chat_runtime import (
    activate_delivery_context,
    capture_delivery_context,
    restore_delivery_context,
    send,
)
from presentation import (
    clear_steering_indicator,
    compacting_indicator,
    typing_loop,
)
from observability import log_agent_activity, log_event
from responses import (
    ContextLengthError,
    TransientResponsesError,
    estimate_response_request_tokens,
    responses_create,
    responses_create_stream,
)
from skills import skill_catalog_instructions


@dataclass(frozen=True)
class RetryableTurnFailure:
    reason: str


def current_session_id() -> str:
    session_id = state.get("current_session_id")
    if not session_id:
        raise RuntimeError("No current session")
    return str(session_id)


def context_tokens() -> int:
    return estimate_response_request_tokens(
        build_active_context(current_session_id()),
        extra_instructions=goal_instructions(),
    )


def estimate_compaction_request(
    items: list[dict[str, Any]],
    instructions: str,
) -> int:
    return estimate_response_request_tokens(
        items,
        tools=None,
        extra_instructions=instructions,
        max_output_tokens=compaction_generation_budget(),
        reasoning_effort=COMPACTION_REASONING_EFFORT,
    )


def _recovery_checkpoint_ref(
    checkpoint: Any,
    *,
    include_id: bool = False,
) -> dict[str, Any] | None:
    """Return bounded recovery metadata safe to inject into a prompt.

    Tool events remain available in the durable transcript/state.  Embedding
    them here made recovery instructions grow without bound, while nesting a
    previous checkpoint under ``resumed_from`` amplified that growth across
    repeated interruptions.
    """
    if not isinstance(checkpoint, dict):
        return None
    result: dict[str, Any] = {}
    if include_id and checkpoint.get("id") is not None:
        result["id"] = checkpoint.get("id")
    for key in ("session_id", "interrupted_at"):
        if checkpoint.get(key) is not None:
            result[key] = checkpoint.get(key)
    events = checkpoint.get("tool_events")
    result["tool_event_count"] = (
        len(events) if isinstance(events, list) else 0
    )
    return result


def _recovery_prompt_payload(
    resumed_agent: Any,
    resumed_subagents: list[Any],
) -> dict[str, Any]:
    return {
        "agent": _recovery_checkpoint_ref(resumed_agent),
        "subagents": [
            ref
            for checkpoint in resumed_subagents
            if (
                ref := _recovery_checkpoint_ref(
                    checkpoint,
                    include_id=True,
                )
            )
            is not None
        ],
    }


async def create_compaction_response(
    items: list[dict[str, Any]],
    instructions: str,
) -> dict[str, Any]:
    return await responses_create(
        items,
        tools=None,
        extra_instructions=instructions,
        max_output_tokens=compaction_generation_budget(),
        reasoning_effort=COMPACTION_REASONING_EFFORT,
    )


async def compact_context(
    chat_id: ConversationId,
    *,
    reason: str = "manual",
) -> None:
    typing_task = asyncio.create_task(typing_loop(chat_id))
    try:
        session_id = current_session_id()
        async with compacting_indicator(chat_id):
            ok = await compact_session(
                session_id,
                estimate_compaction_request=estimate_compaction_request,
                create_compaction_response=create_compaction_response,
                force=False,
                reason=reason,
            )
        if not ok:
            await send(chat_id, "Nothing to compact.")
            return
        stats = session_context_stats(session_id)
        await send(
            chat_id,
            (
                "Checkpoint saved. "
                f"Active native events: {stats['active_native_events']}."
            ),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await send(
            chat_id,
            f"Compact error: {type(exc).__name__}: {exc}",
        )
    finally:
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)


async def maybe_auto_compact(chat_id: ConversationId) -> None:
    budget = context_input_budget()
    if not budget:
        return
    session_id = current_session_id()
    estimated = estimate_response_request_tokens(
        build_active_context(session_id),
        extra_instructions=goal_instructions(),
    )
    if estimated < budget:
        return
    async with compacting_indicator(chat_id):
        await ensure_context_under_budget(
            session_id,
            estimate_active_fn=lambda items: (
                estimate_response_request_tokens(
                    items,
                    extra_instructions=goal_instructions(),
                )
            ),
            estimate_compaction_request=estimate_compaction_request,
            create_compaction_response=create_compaction_response,
            reason="threshold",
            log_source="agent",
        )
    stats = session_context_stats(session_id)
    await send(
        chat_id,
        (
            "Checkpoint saved. "
            f"Active native events: {stats['active_native_events']}."
        ),
    )


def _remove_pending_input(text: str) -> None:
    pending = state.get("pending_inputs")
    if not isinstance(pending, list):
        return
    try:
        pending.remove(text)
    except ValueError:
        pass


async def agent_turn(
    chat_id: ConversationId,
    user_text: str,
    *,
    internal: bool = False,
    source: str | None = None,
    allow_goal_creation: bool = False,
) -> bool | RetryableTurnFailure:
    typing_task = asyncio.create_task(typing_loop(chat_id))
    session_id = current_session_id()
    continue_pending = True
    tool_events: list[dict[str, Any]] = []
    delivery_tokens: list[Any] = []
    resumed_agent = state.get("interrupted_agent")
    resumed_subagents = list(
        state.get("interrupted_subagents", [])
    )
    turn_source = source or (
        "internal" if internal else "user"
    )
    session.goal_creation_authorized = bool(
        allow_goal_creation
        and turn_source == "user"
    )
    user_item = {
        "role": "user",
        "content": user_text,
    }
    append_item(
        session_id,
        user_item,
        source="internal" if turn_source != "user" else "user",
    )
    work = build_active_context(session_id)

    async def apply_pending_steers(
        active_work: list[dict[str, Any]],
        engine_inputs: list[dict[str, Any]],
    ) -> int:
        nonlocal chat_id
        applied = 0
        while True:
            try:
                entry = steer_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if entry.get("status") != "queued":
                    continue
                entry["status"] = "applied"
                session.pending_steers.pop(str(entry["id"]), None)
                text = str(entry["text"])
                next_chat_id = entry["chat_id"]
                next_delivery = entry.get("delivery_context")
                current_delivery = capture_delivery_context(chat_id)
                if (
                    current_delivery is not None
                    and current_delivery != next_delivery
                ):
                    await send(
                        chat_id,
                        "Request superseded by newer steering input.",
                    )
                token = activate_delivery_context(
                    next_chat_id,
                    next_delivery,
                )
                if token is not None:
                    delivery_tokens.append(token)
                chat_id = next_chat_id
                steer_input = {
                    "role": "user",
                    "content": text,
                }
                append_item(
                    session_id,
                    steer_input,
                    source="steer",
                )
                append_meta(
                    session_id,
                    "steer_applied",
                    {"steer_id": entry.get("id")},
                )
                _remove_pending_input(text)
                save_state()
                active_work.append(steer_input)
                engine_inputs.append(steer_input)
                applied += 1
                log_agent_activity("applied steering input")
            finally:
                steer_queue.task_done()
                await clear_steering_indicator(
                    entry["chat_id"],
                    entry.get("message_id"),
                    str(entry.get("id") or ""),
                )
        return applied

    def record_items(
        items: list[dict[str, Any]],
        source: str,
    ) -> None:
        for item in items:
            append_item(session_id, item, source=source)

    try:
        extra_instructions = ""
        if internal:
            extra_instructions = (
                "The current user-role input is an internal agent event, "
                "not a new message authored by the user. Process it and "
                "communicate only the relevant outcome to the user."
            )
        extra_instructions = combine_instructions(
            extra_instructions,
            goal_instructions(),
            await skill_catalog_instructions(),
        )
        if resumed_agent or resumed_subagents:
            recovery = (
                "Recovery context from work interrupted by /stop follows. "
                "Continue safely from the current filesystem state. Do not "
                "assume an in-flight side effect either completed or failed; "
                "inspect before repeating it.\n\n"
                + json.dumps(
                    _recovery_prompt_payload(
                        resumed_agent,
                        resumed_subagents,
                    ),
                    ensure_ascii=False,
                )
            )
            extra_instructions = combine_instructions(
                extra_instructions,
                recovery,
            )
            log_event(
                "agent",
                "checkpoint_loaded",
                {
                    "agent": bool(resumed_agent),
                    "subagents": len(resumed_subagents),
                },
            )

        async def create_response(
            active_work: list[dict[str, Any]],
        ) -> dict[str, Any]:
            budget = context_input_budget()
            estimate: int | None = None
            if budget:
                estimate = estimate_response_request_tokens(
                    active_work,
                    extra_instructions=extra_instructions,
                )
                if estimate >= budget:
                    async with compacting_indicator(chat_id):
                        fitted = await ensure_context_under_budget(
                            session_id,
                            estimate_active_fn=lambda items: (
                                estimate_response_request_tokens(
                                    items,
                                    extra_instructions=(
                                        extra_instructions
                                    ),
                                )
                            ),
                            estimate_compaction_request=(
                                estimate_compaction_request
                            ),
                            create_compaction_response=(
                                create_compaction_response
                            ),
                            reason="threshold",
                            log_source="agent",
                        )
                    active_work[:] = fitted
                    estimate = estimate_response_request_tokens(
                        active_work,
                        extra_instructions=extra_instructions,
                    )

            async def _request(
                items: list[dict[str, Any]],
            ) -> dict[str, Any]:
                return await responses_create_stream(
                    chat_id,
                    items,
                    extra_instructions=extra_instructions,
                )

            async def _compact_forced(
                items: list[dict[str, Any]],
            ) -> list[dict[str, Any]] | None:
                async with compacting_indicator(chat_id):
                    rebuilt = await ensure_context_under_budget(
                        session_id,
                        estimate_active_fn=lambda work: (
                            estimate_response_request_tokens(
                                work,
                                extra_instructions=(
                                    extra_instructions
                                ),
                            )
                        ),
                        estimate_compaction_request=(
                            estimate_compaction_request
                        ),
                        create_compaction_response=(
                            create_compaction_response
                        ),
                        reason="overflow",
                        force=True,
                        include_continuation=True,
                        log_source="agent",
                    )
                log_event(
                    "agent",
                    "context_compacted",
                    {
                        "reason": "overflow",
                        "during_turn": True,
                    },
                )
                return rebuilt

            def _on_overflow(exc: BaseException) -> None:
                log_event(
                    "agent",
                    "context_overflow",
                    {
                        "estimated_tokens": estimate,
                        "prompt_tokens": getattr(
                            exc, "prompt_tokens", None
                        ),
                        "context_tokens": getattr(
                            exc, "context_tokens", None
                        ),
                    },
                )

            return await request_with_checkpoint_retry(
                active_work,
                request_fn=_request,
                compact_fn=_compact_forced,
                overflow_error=ContextLengthError,
                on_overflow=_on_overflow,
            )

        async def execute_call(
            call: dict[str, Any],
        ) -> dict[str, Any]:
            exec_state = dict(state)
            exec_state["session_id"] = session_id
            exec_state["todo_state"] = todo_store_for_execution()
            return await execute_tool_with_approval(
                chat_id,
                call,
                execution_context=exec_state,
            )

        async def compact_work(
            active_work: list[dict[str, Any]],
            *,
            force: bool = False,
        ) -> list[dict[str, Any]] | None:
            reason = "overflow" if force else "threshold"
            if not force:
                budget = context_input_budget()
                if not budget:
                    return None
                est = estimate_response_request_tokens(
                    active_work,
                    extra_instructions=extra_instructions,
                )
                if est < budget:
                    return None
            async with compacting_indicator(chat_id):
                rebuilt = await ensure_context_under_budget(
                    session_id,
                    estimate_active_fn=lambda items: (
                        estimate_response_request_tokens(
                            items,
                            extra_instructions=extra_instructions,
                        )
                    ),
                    estimate_compaction_request=(
                        estimate_compaction_request
                    ),
                    create_compaction_response=(
                        create_compaction_response
                    ),
                    reason=reason,
                    force=force,
                    include_continuation=force,
                    log_source="agent",
                )
            log_event(
                "agent",
                "context_compacted",
                {"reason": reason, "during_turn": True},
            )
            return rebuilt

        def checkpoint(events: list[dict[str, Any]]) -> None:
            tool_events[:] = events

        result = await run_agent(
            work=work,
            create_response=create_response,
            execute_call=execute_call,
            source="agent",
            max_tool_rounds=MAX_TOOL_ROUNDS,
            drain_inputs=apply_pending_steers,
            checkpoint=checkpoint,
            compact_context=compact_work,
            record_items=record_items,
        )
        try:
            await send(chat_id, str(result["text"]))
        except Exception as send_exc:
            log_event(
                "agent",
                "delivery_error",
                {
                    "type": type(send_exc).__name__,
                    "message": str(send_exc),
                },
            )
            if state.get("interrupted_agent") == resumed_agent:
                state["interrupted_agent"] = None
            resumed_ids = {
                str(item.get("id")) for item in resumed_subagents
            }
            state["interrupted_subagents"] = [
                item
                for item in state.get("interrupted_subagents", [])
                if str(item.get("id")) not in resumed_ids
            ]
            save_state()
            return RetryableTurnFailure("response delivery failed")
        if state.get("interrupted_agent") == resumed_agent:
            state["interrupted_agent"] = None
        resumed_ids = {
            str(item.get("id")) for item in resumed_subagents
        }
        state["interrupted_subagents"] = [
            item
            for item in state.get("interrupted_subagents", [])
            if str(item.get("id")) not in resumed_ids
        ]
        save_state()
        await maybe_auto_compact(chat_id)
        return True
    except asyncio.CancelledError:
        continue_pending = False
        checkpoint_payload = {
            "session_id": session_id,
            "interrupted_at": time.time(),
            "tool_events": tool_events,
            "resumed_from": _recovery_checkpoint_ref(
                resumed_agent,
            ),
        }
        state["interrupted_agent"] = checkpoint_payload
        save_state()
        log_event("agent", "checkpoint_saved", checkpoint_payload)
        log_agent_activity("request cancelled")
        raise
    except httpx.HTTPStatusError as exc:
        log_agent_activity("request failed")
        body = exc.response.text[:1200]
        log_event(
            "agent",
            "http_error",
            {
                "status": exc.response.status_code,
                "body": body,
            },
        )
        await send(
            chat_id,
            f"Responses/HTTP error {exc.response.status_code}:\n{body}",
        )
        return False
    except TransientResponsesError as exc:
        log_agent_activity("temporary Responses failure")
        log_event(
            "agent",
            "transient_error",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        await send(
            chat_id,
            f"Temporary Responses failure: {exc}",
        )
        return RetryableTurnFailure("a temporary Responses backend failure")
    except Exception as exc:
        log_agent_activity("request failed")
        log_event(
            "agent",
            "error",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        await send(
            chat_id,
            f"Error: {type(exc).__name__}: {exc}",
        )
        return False
    finally:
        session.goal_creation_authorized = False
        typing_task.cancel()
        await asyncio.gather(typing_task, return_exceptions=True)
        if continue_pending:
            while True:
                try:
                    entry = steer_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                steer_queue.task_done()
                if entry.get("status") != "queued":
                    if entry.get("status") != "forcing":
                        await clear_steering_indicator(
                            entry["chat_id"],
                            entry.get("message_id"),
                            str(entry.get("id") or ""),
                        )
                    continue
                await steer_queue.put(entry)
                break
        else:
            while True:
                try:
                    entry = steer_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                steer_queue.task_done()
                if entry.get("status") == "forcing":
                    continue
                entry["status"] = "cancelled"
                session.pending_steers.pop(str(entry["id"]), None)
                delivery_context = entry.get("delivery_context")
                if delivery_context is not None:
                    token = activate_delivery_context(
                        entry["chat_id"],
                        delivery_context,
                    )
                    try:
                        await send(
                            entry["chat_id"],
                            "Steering cancelled because the run stopped.",
                        )
                    finally:
                        restore_delivery_context(token)
                await clear_steering_indicator(
                    entry["chat_id"],
                    entry.get("message_id"),
                    str(entry.get("id") or ""),
                )
        for token in reversed(delivery_tokens):
            restore_delivery_context(token)
