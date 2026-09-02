"""Agent turn and context compaction entry points."""
import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from raptor.chat.chat_provider import ConversationId

from raptor.state.chat_store import (
    active_checkpoint,
    append_item,
    append_meta,
    event_at_seq,
    next_event_seq,
    reset_model_context,
    session_exists,
)
from config import (
    COMPACTION_REASONING_EFFORT,
    MAX_TOOL_ROUNDS,
    model_compaction_generation_budget,
    model_context_input_budget,
)
from context import (
    build_active_context,
    compact_session,
    ensure_context_under_budget,
    request_with_checkpoint_retry,
)
from engine import (
    assistant_message,
    function_call_output,
    interrupted_tool_result,
    run_agent,
    response_text,
)
from goals import (
    combine_instructions,
    goal_instructions,
    todo_store_for_execution,
)
from raptor.state.session import save_state, state, steer_queue
from raptor.state import session
from approval import execute_tool_with_approval
from raptor.chat.chat_runtime import (
    activate_delivery_context,
    capture_delivery_context,
    detached_delivery_context,
    get_chat_provider,
    restore_delivery_context,
    send,
)
from raptor.chat.presentation import (
    clear_steering_indicator,
    compacting_indicator,
    typing_loop,
)
from observability import log_agent_activity, log_event
from raptor.model.responses import (
    estimate_response_request_tokens,
    model_provider,
    responses_create,
    responses_create_stream,
)
from raptor.model.model_providers import MODEL_CONFIGURATION, ModelTarget
from raptor.model.response_errors import (
    ContextLengthError,
    MalformedToolCallError,
    TransientResponsesError,
)
from skills import skill_catalog_instructions
from raptor.chat.tool_activity import ToolActivitySurface


TURN_ABORTED_GUIDANCE = (
    "The user intentionally interrupted the previous turn. Any unfinished "
    "tool may have partially changed external state; inspect before retrying. "
    "Managed background resources may still be running."
)


def _checkpoint_saved_message(session_id: str) -> str:
    checkpoint = active_checkpoint(session_id)
    seq = checkpoint.get("seq") if checkpoint else None
    if not isinstance(seq, int) or seq <= 0:
        raise RuntimeError("compaction completed without an active checkpoint")
    return "Checkpoint saved"


MALFORMED_TOOL_CALL_MESSAGE = (
    "The model generated an invalid tool call. Nothing was executed, and "
    "the turn was closed safely. Send a new message to continue."
)


def record_turn_interrupted(session_id: str) -> None:
    """Persist a model-visible interruption boundary before releasing a turn."""
    append_item(
        session_id,
        {
            "role": "user",
            "content": (
                "<turn_aborted>\n"
                + TURN_ABORTED_GUIDANCE
                + "\n</turn_aborted>"
            ),
        },
        source="runtime",
    )
    append_meta(session_id, "turn_interrupted", {})


def repair_interrupted_root_turn() -> bool:
    """Close an unclean root turn's durable transcript on process startup."""
    marker = state.get("active_root_turn")
    if not isinstance(marker, dict):
        return False
    session_id = str(marker.get("session_id") or "")
    pending = state.get("pending_delivery")
    if (
        isinstance(pending, dict)
        and str(pending.get("session_id") or "") == session_id
    ):
        delivery_seq = int(pending.get("seq") or 0)
        delivery_event = event_at_seq(session_id, delivery_seq)
        delivery_item = (
            delivery_event.get("item")
            if isinstance(delivery_event, dict)
            else None
        )
        if delivery_event is None:
            session.clear_pending_delivery(session_id, delivery_seq)
        elif (
            delivery_event.get("source") == "assistant"
            and isinstance(delivery_item, dict)
            and delivery_item.get("type") == "message"
        ):
            state["active_root_turn"] = None
            save_state()
            return False
    if session_id and session_exists(session_id):
        unmatched: dict[str, dict[str, Any]] = {}
        for item in build_active_context(session_id):
            item_type = item.get("type")
            call_id = str(item.get("call_id") or "")
            if item_type == "function_call" and call_id:
                unmatched[call_id] = item
            elif item_type == "function_call_output" and call_id:
                unmatched.pop(call_id, None)
        for call in unmatched.values():
            append_item(
                session_id,
                function_call_output(call, interrupted_tool_result()),
                source="tool",
            )
        record_turn_interrupted(session_id)
    state["active_root_turn"] = None
    save_state()
    return True


@dataclass(frozen=True)
class RetryableTurnFailure:
    reason: str


def current_session_id() -> str:
    session_id = state.get("current_session_id")
    if not session_id:
        raise RuntimeError("No current session")
    return str(session_id)


async def flush_pending_delivery(chat_id: ConversationId) -> bool:
    """Retry one archived response before admitting later chat work."""
    reference = state.get("pending_delivery")
    if not isinstance(reference, dict):
        return True
    session_id = str(reference.get("session_id") or "")
    seq = int(reference.get("seq") or 0)
    event = event_at_seq(session_id, seq)
    if event is None:
        session.clear_pending_delivery(session_id, seq)
        log_event(
            "agent",
            "pending_delivery_abandoned",
            {"session_id": session_id, "seq": seq},
        )
        return True
    item = event.get("item") if isinstance(event, dict) else None
    text = response_text({"output": [item]}) if isinstance(item, dict) else ""
    if not text:
        log_event(
            "agent",
            "pending_delivery_invalid",
            {"session_id": session_id, "seq": seq},
        )
        return False
    try:
        with detached_delivery_context(chat_id):
            await send(chat_id, text)
    except Exception as exc:
        log_event(
            "agent",
            "delivery_retry_error",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        return False
    session.clear_pending_delivery(session_id, seq)
    log_event(
        "agent",
        "delivery_retried",
        {"session_id": session_id, "seq": seq},
    )
    return True


def context_tokens() -> int:
    return estimate_response_request_tokens(
        build_active_context(current_session_id()),
        extra_instructions=goal_instructions(),
    )


def _context_window(target: ModelTarget) -> int | None:
    return model_provider(target).settings_for(target.model).context_window


def _input_budget(target: ModelTarget) -> int:
    return model_context_input_budget(_context_window(target))


def _generation_budget(target: ModelTarget) -> int:
    return model_compaction_generation_budget(_context_window(target))


def estimate_compaction_request(
    target: ModelTarget,
    items: list[dict[str, Any]],
    instructions: str,
) -> int:
    return estimate_response_request_tokens(
        items,
        tools=None,
        extra_instructions=instructions,
        max_output_tokens=_generation_budget(target),
        reasoning_effort=COMPACTION_REASONING_EFFORT,
    )


def _subagent_checkpoint_ref(
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
    resumed_subagents: list[Any],
) -> dict[str, Any]:
    return {
        "subagents": [
            ref
            for checkpoint in resumed_subagents
            if (
                ref := _subagent_checkpoint_ref(
                    checkpoint,
                    include_id=True,
                )
            )
            is not None
        ],
    }


async def create_compaction_response(
    target: ModelTarget,
    items: list[dict[str, Any]],
    instructions: str,
) -> dict[str, Any]:
    return await responses_create(
        target,
        items,
        tools=None,
        extra_instructions=instructions,
        max_output_tokens=_generation_budget(target),
        reasoning_effort=COMPACTION_REASONING_EFFORT,
    )


async def compact_context(
    chat_id: ConversationId,
    *,
    reason: str = "manual",
) -> None:
    typing_task = asyncio.create_task(typing_loop(chat_id))
    try:
        target = session.current_model_target()
        compaction_target = MODEL_CONFIGURATION.select_compaction_target(target)
        session_id = current_session_id()
        async with compacting_indicator(chat_id):
            ok = await compact_session(
                session_id,
                estimate_compaction_request=lambda items, instructions: (
                    estimate_compaction_request(
                        compaction_target, items, instructions
                    )
                ),
                create_compaction_response=lambda items, instructions: (
                    create_compaction_response(
                        compaction_target, items, instructions
                    )
                ),
                force=True,
                reason=reason,
                input_budget=_input_budget(compaction_target),
                generation_budget=_generation_budget(compaction_target),
            )
        if not ok:
            await send(chat_id, "Nothing to compact.")
            return
        await send(chat_id, _checkpoint_saved_message(session_id))
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
    target = session.current_model_target()
    compaction_target = MODEL_CONFIGURATION.select_compaction_target(target)
    budget = _input_budget(target)
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
            estimate_compaction_request=lambda items, instructions: (
                estimate_compaction_request(
                    compaction_target, items, instructions
                )
            ),
            create_compaction_response=lambda items, instructions: (
                create_compaction_response(
                    compaction_target, items, instructions
                )
            ),
            reason="threshold",
            log_source="agent",
            input_budget=budget,
            compaction_input_budget=_input_budget(compaction_target),
            generation_budget=_generation_budget(compaction_target),
        )
    await send(chat_id, _checkpoint_saved_message(session_id))


async def agent_turn(
    chat_id: ConversationId,
    user_text: str,
    *,
    internal: bool = False,
    source: str | None = None,
    allow_goal_creation: bool = False,
    input_recorded: bool = False,
    source_message_id: int | str | None = None,
) -> bool | RetryableTurnFailure:
    typing_task: asyncio.Task[None] | None = asyncio.create_task(
        typing_loop(chat_id)
    )

    async def pause_typing() -> None:
        nonlocal typing_task
        task = typing_task
        typing_task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def resume_typing() -> None:
        nonlocal typing_task
        if typing_task is None:
            typing_task = asyncio.create_task(typing_loop(chat_id))

    target = session.current_model_target()
    compaction_target = MODEL_CONFIGURATION.select_compaction_target(target)
    session_id = current_session_id()
    continue_pending = True
    response_delivered = False
    delivery_tokens: list[Any] = []
    final_output_seq: int | None = None
    resumed_subagents = list(
        state.get("interrupted_subagents", [])
    )
    tool_activity = ToolActivitySurface(chat_id)
    turn_source = source or (
        "internal" if internal else "user"
    )
    runtime = session.current_runtime()
    runtime.goal_creation_authorized = bool(
        allow_goal_creation
        and turn_source == "user"
    )
    user_item = {
        "role": "user",
        "content": user_text,
    }
    if not input_recorded:
        append_item(
            session_id,
            user_item,
            source=turn_source,
            data=(
                {
                    "chat_message": {
                        "conversation_id": get_chat_provider().encode_conversation_id(
                            chat_id
                        ),
                        "message_id": source_message_id,
                    }
                }
                if turn_source == "user" and source_message_id is not None
                else None
            ),
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
                session.persist_steer_handoff(entry)
                append_meta(
                    session_id,
                    "steer_applied",
                    {"steer_id": entry.get("id")},
                )
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

    def prepare_delivery(item: dict[str, Any]) -> int:
        delivery_seq = next_event_seq(session_id)
        session.set_pending_delivery(session_id, delivery_seq)
        try:
            append_item(
                session_id,
                item,
                source="assistant",
                expected_seq=delivery_seq,
            )
        except BaseException:
            session.clear_pending_delivery(session_id, delivery_seq)
            raise
        return delivery_seq

    def record_terminal_items(
        items: list[dict[str, Any]],
        text: str,
    ) -> None:
        nonlocal final_output_seq
        messages = [item for item in items if item.get("type") == "message"]
        if not messages:
            raise RuntimeError("Agent terminal output contains no message")
        delivery_item = (
            messages[0]
            if len(messages) == 1
            else assistant_message(text)
        )
        message_recorded = False
        for item in items:
            if item.get("type") == "message":
                if not message_recorded:
                    final_output_seq = prepare_delivery(delivery_item)
                    message_recorded = True
                continue
            append_item(session_id, item, source="assistant")

    async def deliver_terminal_failure(
        message: str,
        *,
        reset_context: bool = False,
    ) -> None:
        await tool_activity.clear()
        await pause_typing()
        delivery_seq = prepare_delivery(assistant_message(message))
        if reset_context:
            reset_model_context(
                session_id,
                through_seq=delivery_seq,
            )
        try:
            await send(chat_id, message)
        except Exception as exc:
            log_event(
                "agent",
                "delivery_error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        else:
            session.clear_pending_delivery(session_id, delivery_seq)

    try:
        extra_instructions = ""
        if internal:
            extra_instructions = (
                "The current user-role input is an internal Raptor event, "
                "not a message authored by the user. Process it and "
                "communicate only the relevant outcome."
            )
        extra_instructions = combine_instructions(
            extra_instructions,
            goal_instructions(),
            await skill_catalog_instructions(),
        )
        if resumed_subagents:
            recovery = (
                "Recovery context from interrupted subagents follows. "
                "Continue safely from the current filesystem state. Do not "
                "assume an in-flight side effect either completed or failed; "
                "inspect before repeating it.\n\n"
                + json.dumps(
                    _recovery_prompt_payload(
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
                    "subagents": len(resumed_subagents),
                },
            )

        async def create_response(
            active_work: list[dict[str, Any]],
        ) -> dict[str, Any]:
            budget = _input_budget(target)
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
                                lambda items, instructions: (
                                    estimate_compaction_request(
                                        compaction_target, items, instructions
                                    )
                                )
                            ),
                            create_compaction_response=(
                                lambda items, instructions: (
                                    create_compaction_response(
                                        compaction_target, items, instructions
                                    )
                                )
                            ),
                            reason="threshold",
                            include_continuation=True,
                            log_source="agent",
                            input_budget=budget,
                            compaction_input_budget=(
                                _input_budget(compaction_target)
                            ),
                            generation_budget=(
                                _generation_budget(compaction_target)
                            ),
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
                    target,
                    chat_id,
                    items,
                    extra_instructions=extra_instructions,
                    on_tool_call=(
                        tool_activity.stream
                        if tool_activity is not None
                        else None
                    ),
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
                            lambda items, instructions: (
                                estimate_compaction_request(
                                    compaction_target, items, instructions
                                )
                            )
                        ),
                        create_compaction_response=(
                            lambda items, instructions: (
                                create_compaction_response(
                                    compaction_target, items, instructions
                                )
                            )
                        ),
                        reason="overflow",
                        force=True,
                        include_continuation=True,
                        log_source="agent",
                        input_budget=_input_budget(target),
                        compaction_input_budget=(
                            _input_budget(compaction_target)
                        ),
                        generation_budget=(
                            _generation_budget(compaction_target)
                        ),
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
            await pause_typing()
            resume_after_tool = True
            exec_state = dict(state)
            exec_state["session_id"] = session_id
            exec_state["model_target"] = target.to_dict()
            exec_state["todo_state"] = todo_store_for_execution()
            try:
                result = await execute_tool_with_approval(
                    chat_id,
                    call,
                    execution_context=exec_state,
                    tool_activity=tool_activity,
                )
                return result
            except asyncio.CancelledError:
                resume_after_tool = False
                raise
            finally:
                if resume_after_tool:
                    resume_typing()

        async def compact_work(
            active_work: list[dict[str, Any]],
            *,
            force: bool = False,
        ) -> list[dict[str, Any]] | None:
            reason = "overflow" if force else "threshold"
            if not force:
                budget = _input_budget(target)
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
                        lambda items, instructions: (
                            estimate_compaction_request(
                                compaction_target, items, instructions
                            )
                        )
                    ),
                    create_compaction_response=(
                        lambda items, instructions: (
                            create_compaction_response(
                                compaction_target, items, instructions
                            )
                        )
                    ),
                    reason=reason,
                    force=force,
                    include_continuation=True,
                    log_source="agent",
                    input_budget=_input_budget(target),
                    compaction_input_budget=_input_budget(compaction_target),
                    generation_budget=_generation_budget(compaction_target),
                )
            log_event(
                "agent",
                "context_compacted",
                {"reason": reason, "during_turn": True},
            )
            return rebuilt

        result = await run_agent(
            work=work,
            create_response=create_response,
            execute_call=execute_call,
            source="agent",
            max_tool_rounds=MAX_TOOL_ROUNDS,
            drain_inputs=apply_pending_steers,
            compact_context=compact_work,
            record_items=record_items,
            record_terminal_items=record_terminal_items,
            report_tool_result=tool_activity.finished,
        )
        if final_output_seq is None:
            raise RuntimeError("Final response was not archived")
        await tool_activity.clear()
        await pause_typing()
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
        session.clear_pending_delivery(session_id, final_output_seq)
        response_delivered = True
        resumed_ids = {
            str(item.get("id")) for item in resumed_subagents
        }
        state["interrupted_subagents"] = [
            item
            for item in state.get("interrupted_subagents", [])
            if str(item.get("id")) not in resumed_ids
        ]
        save_state()
        try:
            await maybe_auto_compact(chat_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                "agent",
                "post_delivery_compaction_error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        return True
    except asyncio.CancelledError:
        continue_pending = False
        if not response_delivered:
            record_turn_interrupted(session_id)
        log_agent_activity("request cancelled")
        raise
    except MalformedToolCallError as exc:
        log_agent_activity("model generated malformed tool arguments")
        log_event(
            "agent",
            "malformed_tool_call",
            {"message": str(exc)},
        )
        await deliver_terminal_failure(
            MALFORMED_TOOL_CALL_MESSAGE,
            reset_context=True,
        )
        return RetryableTurnFailure("an invalid model tool call")
    except httpx.HTTPStatusError as exc:
        log_agent_activity("request failed")
        body = exc.response.text[:1200]
        log_event(
            "agent",
            "http_error",
            {
                "status": exc.response.status_code,
                "body_chars": len(exc.response.text),
            },
        )
        await deliver_terminal_failure(
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
        await deliver_terminal_failure(
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
        await deliver_terminal_failure(
            f"Error: {type(exc).__name__}: {exc}",
        )
        return False
    finally:
        runtime.goal_creation_authorized = False
        await tool_activity.clear()
        await pause_typing()
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
            from steering import cancel_unforced_steers

            await cancel_unforced_steers()
        for token in reversed(delivery_tokens):
            restore_delivery_context(token)
