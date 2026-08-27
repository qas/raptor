"""Recursive foreground and background agent orchestration."""
import asyncio
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

import session
from activity import (
    append_subagent_activity_input,
    delete_subagent_activity,
    finish_subagent_activity,
    open_subagent_activity,
    publish_subagent_activity,
    publish_subagent_response,
)
from chat_provider import ConversationId
from chat_store import (
    append_item,
    create_session,
    reset_model_context,
)
from config import (
    BASE_INSTRUCTIONS,
    COMPACTION_REASONING_EFFORT,
    MAX_BACKGROUND_SUBAGENTS,
    MAX_SUBAGENT_DEPTH,
    MAX_SUBAGENT_PENDING_INPUTS,
    MAX_SUBAGENT_RECORDS,
    MAX_SUBAGENT_TOOL_EVENTS,
    MAX_TOOL_OUTPUT,
    MAX_TOOL_ROUNDS,
    TOOLS,
    model_compaction_generation_budget,
    model_context_input_budget,
)
from context import (
    build_active_context,
    checkpoint_continuation_input,
    compact_session,
    ensure_context_under_budget,
    request_with_checkpoint_retry,
)
from engine import assistant_message, estimate_tokens, run_agent
from observability import log_event
from response_errors import (
    ContextLengthError,
    MalformedToolCallError,
)
from responses import (
    ResponsesStreamReplayGuard,
    parse_http_response_error,
    retry_transient_response,
    stream_response_payload,
    validate_chronological_input,
    model_provider,
)
from model_providers import MODEL_CONFIGURATION, ModelTarget
from session import (
    bounded_interrupted_subagents,
    prune_subagent_records,
    save_state,
    state,
)
from skills import skill_catalog_instructions

_background_reservations = 0


def _background_subagent_count() -> int:
    return _background_reservations + sum(
        len(runtime.subagent_tasks)
        for runtime in session.all_chat_runtimes()
    )


def _background_capacity_error() -> dict[str, Any] | None:
    pending_completions = sum(
        bool(item.get("completion_pending"))
        for runtime in session.all_chat_runtimes()
        for item in runtime.subagent_records.values()
    )
    if pending_completions >= MAX_SUBAGENT_RECORDS:
        return {
            "ok": False,
            "status": "capacity_reached",
            "error": (
                "Pending subagent completions must be delivered before "
                "more background subagents can start"
            ),
        }
    if _background_subagent_count() >= MAX_BACKGROUND_SUBAGENTS:
        return {
            "ok": False,
            "status": "capacity_reached",
            "error": (
                "Background subagent capacity reached "
                f"({MAX_BACKGROUND_SUBAGENTS})"
            ),
        }
    return None


def _bounded_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for event in events[-MAX_SUBAGENT_TOOL_EVENTS:]:
        call = event.get("call")
        result = event.get("result")
        projected.append(
            {
                "call": {
                    "name": call.get("name"),
                    "call_id": call.get("call_id"),
                }
                if isinstance(call, dict)
                else None,
                "status": event.get("status"),
                "result": {
                    "ok": result.get("ok"),
                    "status": result.get("status"),
                    "has_error": bool(result.get("error")),
                }
                if isinstance(result, dict)
                else None,
            }
        )
    return projected


def _bounded_record_text(value: Any) -> str:
    return str(value)[:MAX_TOOL_OUTPUT]


def subagent_tools(
    allow_subagents: bool,
    depth: int,
) -> list[dict[str, Any]]:
    excluded = {
        "cancel",
        "get_goal",
        "update_goal",
        "set_goal",
    }
    if not (
        allow_subagents
        and depth < MAX_SUBAGENT_DEPTH
    ):
        excluded.add("subagent")
    return [
        tool
        for tool in TOOLS
        if tool.get("name") not in excluded
    ]


def subagent_instructions(
    allow_subagents: bool,
) -> str:
    delegation = (
        "You may use the subagent tool because the user explicitly allowed "
        "nested delegation."
        if allow_subagents
        else "You may not delegate to another subagent."
    )
    return (
        BASE_INSTRUCTIONS
        + "\n\nYou are an isolated subagent working for a parent agent. "
        "Complete only the delegated task. Never communicate through the chat "
        "provider "
        "or address the user directly. Return a concise, evidence-based result "
        "to the parent agent. Your conversation context and todos are isolated, "
        "but filesystem and shell side effects occur in the shared workspace. "
        + delegation
    )


def record_model_target(record: dict[str, Any]) -> ModelTarget:
    return ModelTarget.from_value(record.get("model_target"))


def target_input_budget(target: ModelTarget) -> int:
    window = model_provider(target).settings_for(target.model).context_window
    return model_context_input_budget(window)


def target_generation_budget(target: ModelTarget) -> int:
    window = model_provider(target).settings_for(target.model).context_window
    return model_compaction_generation_budget(window)


def _recovery_prompt_payload(
    recovery_context: Any,
) -> dict[str, Any] | None:
    """Keep subagent recovery instructions bounded.

    Raw tool events are already durable in the subagent transcript and state;
    only their count is needed in every resumed model request.
    """
    if not isinstance(recovery_context, dict):
        return None
    result = {
        key: recovery_context.get(key)
        for key in ("status", "last_task", "error")
        if recovery_context.get(key) is not None
    }
    events = recovery_context.get("tool_events")
    result["tool_event_count"] = (
        len(events) if isinstance(events, list) else 0
    )
    return result


def build_subagent_payload(
    target: ModelTarget,
    work: list[dict[str, Any]],
    *,
    allow_subagents: bool,
    depth: int,
    tools: list[dict[str, Any]] | None = None,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    validate_chronological_input(work)
    instructions = subagent_instructions(allow_subagents)
    if extra_instructions:
        instructions += "\n\n" + extra_instructions
    settings = model_provider(target).settings_for(target.model)
    payload: dict[str, Any] = {
        "model": target.model,
        "input": work,
        "instructions": instructions,
        "stream": stream,
    }
    selected_tools = (
        subagent_tools(allow_subagents, depth)
        if tools is None
        else tools
    )
    if selected_tools:
        payload["tools"] = selected_tools
        payload["parallel_tool_calls"] = False
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    reasoning_effort = reasoning_effort or settings.reasoning_effort
    if reasoning_effort is not None or reasoning_summary is not None:
        payload["reasoning"] = {}
        if reasoning_effort is not None:
            payload["reasoning"]["effort"] = reasoning_effort
        if reasoning_summary is not None:
            payload["reasoning"]["summary"] = reasoning_summary
    return payload


def estimate_subagent_request_tokens(
    target: ModelTarget,
    work: list[dict[str, Any]],
    *,
    allow_subagents: bool,
    depth: int,
    tools: list[dict[str, Any]] | None = None,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> int:
    return estimate_tokens(
        build_subagent_payload(
            target,
            work,
            allow_subagents=allow_subagents,
            depth=depth,
            tools=tools,
            extra_instructions=extra_instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
    )


async def _create_subagent_response_once(
    target: ModelTarget,
    work: list[dict[str, Any]],
    *,
    agent_id: str,
    allow_subagents: bool,
    depth: int,
    tools: list[dict[str, Any]] | None = None,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    on_reasoning_summary: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    streaming = on_text is not None or on_reasoning_summary is not None
    provider = model_provider(target)
    settings = provider.settings_for(target.model)
    payload = build_subagent_payload(
        target,
        work,
        allow_subagents=allow_subagents,
        depth=depth,
        tools=tools,
        extra_instructions=extra_instructions,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=(
            reasoning_summary or settings.reasoning_summary
            if streaming
            else None
        ),
        stream=streaming,
    )
    if streaming:
        return await stream_response_payload(
            url=f"{provider.base_url}/responses",
            headers=provider.headers(),
            payload=payload,
            on_text=on_text,
            on_reasoning_summary=on_reasoning_summary,
            log_source="subagent",
            log_data={"agent_id": agent_id},
        )
    response = await session.responses.post(
        f"{provider.base_url}/responses",
        headers=provider.headers(),
        json=payload,
        timeout=None,
    )
    if response.is_error:
        log_event(
            "subagent",
            "http_error",
            {
                "agent_id": agent_id,
                "status": response.status_code,
                "body_chars": len(response.text),
            },
        )
        classified = parse_http_response_error(response)
        if classified:
            raise classified
    response.raise_for_status()
    return response.json()


async def create_subagent_response(
    target: ModelTarget,
    work: list[dict[str, Any]],
    *,
    agent_id: str,
    allow_subagents: bool,
    depth: int,
    tools: list[dict[str, Any]] | None = None,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    on_reasoning_summary: Callable[[str], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    replay_guard = ResponsesStreamReplayGuard()

    async def request() -> dict[str, Any]:
        try:
            return await _create_subagent_response_once(
                target,
                work,
                agent_id=agent_id,
                allow_subagents=allow_subagents,
                depth=depth,
                tools=tools,
                extra_instructions=extra_instructions,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                on_text=replay_guard.wrap(on_text),
                on_reasoning_summary=replay_guard.wrap(on_reasoning_summary),
            )
        except Exception as exc:
            replay_guard.reject_unsafe_replay(
                exc,
                operation="Subagent stream",
            )
            raise

    provider = model_provider(target)
    return await retry_transient_response(
        request,
        operation="Subagent Responses request",
        log_data={"agent_id": agent_id},
        max_retries=provider.request_max_retries,
        retry_base_seconds=provider.retry_base_seconds,
    )


async def compact_subagent_session(
    record: dict[str, Any],
    *,
    allow_subagents: bool,
    depth: int,
    reason: str = "threshold",
) -> bool:
    session_id = str(record["session_id"])
    agent_id = str(record["id"])
    target = record_model_target(record)
    generation_budget = target_generation_budget(target)
    input_budget = target_input_budget(target)

    def estimate(items, instructions):
        return estimate_subagent_request_tokens(
            target,
            items,
            allow_subagents=allow_subagents,
            depth=depth,
            tools=[],
            extra_instructions=instructions,
            max_output_tokens=generation_budget,
            reasoning_effort=COMPACTION_REASONING_EFFORT,
        )

    async def create(items, instructions):
        return await create_subagent_response(
            target,
            items,
            agent_id=agent_id,
            allow_subagents=allow_subagents,
            depth=depth,
            tools=[],
            extra_instructions=instructions,
            max_output_tokens=generation_budget,
            reasoning_effort=COMPACTION_REASONING_EFFORT,
        )

    return await compact_session(
        session_id,
        estimate_compaction_request=estimate,
        create_compaction_response=create,
        force=reason == "overflow",
        reason=reason,
        input_budget=input_budget,
        generation_budget=generation_budget,
    )


async def run_subagent(
    *,
    agent_id: str,
    chat_id: ConversationId,
    depth: int,
    allow_subagents: bool,
) -> str:
    from approval import execute_tool_with_approval
    record = session.subagent_records[agent_id]
    target = record_model_target(record)
    input_budget = target_input_budget(target)
    generation_budget = target_generation_budget(target)
    session_id = str(record["session_id"])
    work = build_active_context(session_id)
    if not work:
        raise RuntimeError("Subagent session has no input")
    record.setdefault("todos", [])
    record["subagents_allowed"] = allow_subagents
    previous_events = list(record.get("tool_events", []))
    recovery_context = record.get("recovery_context")
    skills_instructions = await skill_catalog_instructions()

    async def drain_inputs(
        active_work: list[dict[str, Any]],
        turn_inputs: list[dict[str, Any]],
    ) -> int:
        applied = 0
        pending = record.setdefault("pending_inputs", [])
        while pending:
            text = str(pending.pop(0))
            steer_input = {"role": "user", "content": text}
            append_item(session_id, steer_input, source="steer")
            active_work.append(steer_input)
            turn_inputs.append(steer_input)
            applied += 1
        if applied:
            save_state()
        return applied

    def record_items(
        items: list[dict[str, Any]],
        source: str,
    ) -> None:
        for item in items:
            append_item(session_id, item, source=source)

    async def create_response(
        active_work: list[dict[str, Any]],
    ) -> dict[str, Any]:
        budget = input_budget
        estimate: int | None = None
        recovery_inst = (
            (
                "Recovery context from interrupted work follows. "
                "Continue safely from the shared workspace state. "
                "Do not assume an in-flight side effect completed "
                "or failed; inspect before repeating it.\n\n"
                + json.dumps(
                    _recovery_prompt_payload(recovery_context),
                    ensure_ascii=False,
                )
            )
            if recovery_context
            else ""
        )
        if skills_instructions:
            recovery_inst = "\n\n".join(
                part
                for part in (recovery_inst, skills_instructions)
                if part
            )
        if budget:
            estimate = estimate_subagent_request_tokens(
                target,
                active_work,
                allow_subagents=allow_subagents,
                depth=depth,
                extra_instructions=recovery_inst,
            )
            if estimate >= budget:
                fitted = await ensure_context_under_budget(
                    session_id,
                    estimate_active_fn=lambda items: (
                        estimate_subagent_request_tokens(
                            target,
                            items,
                            allow_subagents=allow_subagents,
                            depth=depth,
                            extra_instructions=recovery_inst,
                        )
                    ),
                    estimate_compaction_request=lambda items, instructions: (
                        estimate_subagent_request_tokens(
                            target,
                            items,
                            allow_subagents=allow_subagents,
                            depth=depth,
                            tools=[],
                            extra_instructions=instructions,
                            max_output_tokens=generation_budget,
                            reasoning_effort=COMPACTION_REASONING_EFFORT,
                        )
                    ),
                    create_compaction_response=lambda items, instructions: (
                        create_subagent_response(
                            target,
                            items,
                            agent_id=agent_id,
                            allow_subagents=allow_subagents,
                            depth=depth,
                            tools=[],
                            extra_instructions=instructions,
                            max_output_tokens=generation_budget,
                            reasoning_effort=COMPACTION_REASONING_EFFORT,
                        )
                    ),
                    reason="threshold",
                    log_source="subagent",
                    input_budget=budget,
                    generation_budget=generation_budget,
                )
                active_work[:] = fitted
                estimate = estimate_subagent_request_tokens(
                    target,
                    active_work,
                    allow_subagents=allow_subagents,
                    depth=depth,
                    extra_instructions=recovery_inst,
                )

        async def _request(items):
            if not record.get("activity_surface_id"):
                return await create_subagent_response(
                    target,
                    items,
                    agent_id=agent_id,
                    allow_subagents=allow_subagents,
                    depth=depth,
                    extra_instructions=recovery_inst,
                )

            publish_subagent_response(
                record,
                reasoning_summary="",
                reply="",
            )

            async def publish_text(text: str) -> None:
                publish_subagent_response(record, reply=text)

            async def publish_reasoning(summary: str) -> None:
                publish_subagent_response(
                    record,
                    reasoning_summary=summary,
                )

            return await create_subagent_response(
                target,
                items,
                agent_id=agent_id,
                allow_subagents=allow_subagents,
                depth=depth,
                extra_instructions=recovery_inst,
                on_text=publish_text,
                on_reasoning_summary=publish_reasoning,
            )

        async def _compact_forced(items):
            rebuilt = await ensure_context_under_budget(
                session_id,
                estimate_active_fn=lambda work: (
                    estimate_subagent_request_tokens(
                        target,
                        work,
                        allow_subagents=allow_subagents,
                        depth=depth,
                        extra_instructions=recovery_inst,
                    )
                ),
                estimate_compaction_request=lambda work, instructions: (
                    estimate_subagent_request_tokens(
                        target,
                        work,
                        allow_subagents=allow_subagents,
                        depth=depth,
                        tools=[],
                        extra_instructions=instructions,
                        max_output_tokens=generation_budget,
                        reasoning_effort=COMPACTION_REASONING_EFFORT,
                    )
                ),
                create_compaction_response=lambda work, instructions: (
                    create_subagent_response(
                        target,
                        work,
                        agent_id=agent_id,
                        allow_subagents=allow_subagents,
                        depth=depth,
                        tools=[],
                        extra_instructions=instructions,
                        max_output_tokens=generation_budget,
                        reasoning_effort=COMPACTION_REASONING_EFFORT,
                    )
                ),
                reason="overflow",
                force=True,
                include_continuation=True,
                log_source="subagent",
                input_budget=input_budget,
                generation_budget=generation_budget,
            )
            return rebuilt

        def _on_overflow(exc: BaseException) -> None:
            log_event(
                "subagent",
                "context_overflow",
                {
                    "agent_id": agent_id,
                    "estimated_tokens": estimate,
                    "prompt_tokens": getattr(exc, "prompt_tokens", None),
                    "context_tokens": getattr(exc, "context_tokens", None),
                },
            )

        return await request_with_checkpoint_retry(
            active_work,
            request_fn=_request,
            compact_fn=_compact_forced,
            overflow_error=ContextLengthError,
            on_overflow=_on_overflow,
        )

    async def execute_call(call: dict[str, Any]) -> dict[str, Any]:
        # Keep chat_history scoped to this subagent transcript.
        tool_context = dict(record)
        tool_context["session_id"] = session_id
        tool_context["model_target"] = target.to_dict()
        tool_context["todo_state"] = record
        return await execute_tool_with_approval(
            chat_id,
            call,
            execution_context=tool_context,
        )

    async def compact_work(
        active_work: list[dict[str, Any]],
        *,
        force: bool = False,
    ) -> list[dict[str, Any]] | None:
        reason = "overflow" if force else "threshold"
        if not force:
            budget = input_budget
            if not budget:
                return None
            est = estimate_subagent_request_tokens(
                target,
                active_work,
                allow_subagents=allow_subagents,
                depth=depth,
            )
            if est < budget:
                return None
        ok = await compact_subagent_session(
            record,
            allow_subagents=allow_subagents,
            depth=depth,
            reason=reason,
        )
        if not ok:
            return None
        rebuilt = build_active_context(session_id)
        if reason == "overflow":
            rebuilt.extend(checkpoint_continuation_input())
        return rebuilt

    def checkpoint(events: list[dict[str, Any]]) -> None:
        record["tool_events"] = _bounded_tool_events(
            previous_events + list(events)
        )
        save_state()

    try:
        result = await run_agent(
            work=work,
            create_response=create_response,
            execute_call=execute_call,
            source="subagent",
            agent_id=agent_id,
            max_tool_rounds=MAX_TOOL_ROUNDS,
            drain_inputs=drain_inputs,
            checkpoint=checkpoint,
            compact_context=compact_work,
            record_items=record_items,
            report_activity=lambda detail: publish_subagent_activity(
                record,
                detail,
            ),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        message = _bounded_record_text(
            f"Subagent failed: {type(exc).__name__}: {exc}"
        )
        outcome = append_item(
            session_id,
            assistant_message(message),
            source="assistant",
        )
        if isinstance(exc, MalformedToolCallError):
            reset_model_context(
                session_id,
                through_seq=int(outcome["seq"]),
            )
        raise
    record["tool_events"] = _bounded_tool_events(
        previous_events + list(result["tool_events"])
    )
    record["recovery_context"] = None
    save_state()
    if record.get("pending_inputs"):
        text = str(record["pending_inputs"].pop(0))
        append_item(
            session_id,
            {"role": "user", "content": text},
            source="steer",
        )
        save_state()
        return await run_subagent(
            agent_id=agent_id,
            chat_id=chat_id,
            depth=depth,
            allow_subagents=allow_subagents,
        )
    return str(result["text"])


def new_record(
    task: str,
    *,
    target: ModelTarget,
    chat_id: ConversationId,
    depth: int,
    background: bool,
    allow_subagents: bool,
    parent_session_id: str | None,
    root_session_id: str | None,
) -> dict[str, Any]:
    agent_id = secrets.token_hex(4)
    session_id = create_session(
        kind="subagent",
        chat_key=session.current_runtime().key,
        agent_id=agent_id,
        parent_session_id=(
            str(parent_session_id)
            if parent_session_id
            else None
        ),
        model_target=target.to_dict(),
    )
    append_item(
        session_id,
        {"role": "user", "content": task},
        source="delegation",
    )
    return {
        "id": agent_id,
        "chat_key": session.current_runtime().key,
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "root_session_id": root_session_id,
        "model_target": target.to_dict(),
        "task": task,
        "last_task": task,
        "task_count": 1,
        "todos": [],
        "pending_inputs": [],
        "recovery_context": None,
        "chat_id": chat_id,
        "depth": depth,
        "background": background,
        "allow_subagents": allow_subagents,
        "status": "running",
        "started_at": int(time.time()),
        "completed_at": None,
        "result": None,
        "error": None,
        "tool_events": [],
        "notify_completion": background,
        "completion_pending": False,
        "completion_notified_at": None,
        "completion_attempts": 0,
        "run_generation": 1,
        "activity_finished_generation": 0,
        "activity_result_delivered": False,
        "activity_surface_id": None,
    }


def lifecycle_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "agent_id":
            record.get("id"),
        "status":
            record.get("status"),
        "background":
            record.get(
                "background"
            ),
        "depth":
            record.get("depth"),
        "model_target": record.get("model_target"),
        "started_at":
            record.get(
                "started_at"
            ),
        "completed_at":
            record.get(
                "completed_at"
            ),
        "task_count": record.get("task_count", 1),
        "tool_event_count":
            len(
                record.get(
                    "tool_events",
                    [],
                )
            ),
        "todo_count":
            len(
                record.get(
                    "todos",
                    [],
                )
            ),
        "completion_pending":
            record.get(
                "completion_pending",
                False,
            ),
        "has_error": bool(record.get("error")),
    }


def subagent_summaries() -> list[
    dict[str, Any]
]:
    current = state.get("current_session_id")
    rows: list[dict[str, Any]] = []
    for record in session.subagent_records.values():
        if (
            current
            and record.get("parent_session_id") not in {
                None,
                current,
            }
        ):
            # Historical sessions stay discoverable via chat_history.
            continue
        rows.append(
            {
                "id": record.get("id"),
                "status": record.get("status"),
                "task": record.get("task"),
                "last_task": record.get("last_task"),
                "depth": record.get("depth"),
                "model_target": record.get("model_target"),
                "background": record.get("background"),
                "session_id": record.get("session_id"),
                "parent_session_id": record.get(
                    "parent_session_id"
                ),
                "started_at": record.get("started_at"),
                "completed_at": record.get("completed_at"),
            }
        )
    return rows


def subagent_status(record: dict[str, Any]) -> dict[str, Any]:
    """Project stable public state without exposing a child's private context."""
    return {
        "id": record.get("id"),
        "status": record.get("status"),
        "task": record.get("task"),
        "last_task": record.get("last_task"),
        "task_count": record.get("task_count", 1),
        "depth": record.get("depth"),
        "background": record.get("background"),
        "model_target": record.get("model_target"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "result": record.get("result"),
        "error": record.get("error"),
    }


def continue_record(
    record: dict[str, Any],
    task: str,
    *,
    chat_id: ConversationId,
    depth: int,
    background: bool,
    allow_subagents: bool,
) -> None:
    if record.get(
        "status"
    ) in {
        "cancelled",
        "interrupted",
    }:
        record[
            "recovery_context"
        ] = {
            "status":
                record.get(
                    "status"
                ),
            "last_task":
                record.get(
                    "last_task"
                ),
            "error":
                record.get(
                    "error"
                ),
            "tool_events":
                record.get(
                    "tool_events",
                    [],
                ),
        }
    for pending_text in list(record.get("pending_inputs") or []):
        append_item(
            str(record["session_id"]),
            {"role": "user", "content": str(pending_text)},
            source="steer",
        )
    record["pending_inputs"] = []
    record["last_task"] = task
    record["task_count"] = int(record.get("task_count") or 1) + 1
    append_item(
        str(record["session_id"]),
        {
            "role": "user",
            "content": task,
        },
        source="delegation",
    )
    record["chat_id"] = chat_id
    record["chat_key"] = session.current_runtime().key
    record["depth"] = depth
    record["background"] = background
    record[
        "allow_subagents"
    ] = allow_subagents
    record["status"] = "running"
    record["started_at"] = int(
        time.time()
    )
    record["completed_at"] = None
    record["result"] = None
    record["error"] = None
    record[
        "notify_completion"
    ] = background
    record[
        "completion_pending"
    ] = False
    record[
        "completion_notified_at"
    ] = None
    record[
        "completion_attempts"
    ] = 0
    record["run_generation"] = max(
        1,
        int(record.get("run_generation") or 1) + 1,
    )
    record["activity_result_delivered"] = False
    save_state()


def save_interrupted_subagent(
    record: dict[str, Any],
) -> None:
    checkpoint = {
        "id": record.get("id"),
        "session_id": record.get("session_id"),
        "interrupted_at": time.time(),
        "tool_events": _bounded_tool_events(
            list(record.get("tool_events") or [])
        ),
    }
    interrupted = list(
        state.get(
            "interrupted_subagents",
            [],
        )
    )
    interrupted = [
        item
        for item in interrupted
        if item.get("id")
        != checkpoint.get("id")
    ]
    interrupted.append(
        checkpoint
    )
    state["interrupted_subagents"] = bounded_interrupted_subagents(interrupted)
    save_state()
    log_event(
        "subagent",
        "checkpoint_saved",
        lifecycle_record(
            record
        ),
    )


async def run_background_subagent(
    record: dict[str, Any],
) -> None:
    agent_id = str(
        record["id"]
    )
    owner_task = asyncio.current_task()
    generation = max(1, int(record.get("run_generation") or 1))
    try:
        result = await run_subagent(
            agent_id=agent_id,
            chat_id=record["chat_id"],
            depth=int(
                record["depth"]
            ),
            allow_subagents=bool(
                record["allow_subagents"]
            ),
        )
        record["result"] = _bounded_record_text(result)
        record["status"] = "completed"
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["error"] = (
            "Subagent was cancelled"
        )
        record["completed_at"] = int(
            time.time()
        )
        save_interrupted_subagent(
            record
        )
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = _bounded_record_text(
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            if int(record.get("run_generation") or 1) == generation:
                record["completed_at"] = int(time.time())
                should_notify = bool(record.get("notify_completion", True))
                record["completion_pending"] = should_notify
                save_state()
                log_event("subagent", "completed", lifecycle_record(record))
                if should_notify:
                    _queue_subagent_completion(record)
                await finish_subagent_activity(
                    record,
                    expected_generation=generation,
                )
        finally:
            if session.subagent_tasks.get(agent_id) is owner_task:
                session.subagent_tasks.pop(agent_id, None)


async def delete_subagent_record(agent_id: str) -> dict[str, Any]:
    """Delete one terminal subagent and its provider-owned surface."""
    record = session.subagent_records.get(agent_id)
    if record is None:
        return {"ok": False, "error": f"subagent {agent_id} not found"}
    task = session.subagent_tasks.get(agent_id)
    if record.get("status") == "running" or (
        task is not None and not task.done()
    ):
        return {
            "ok": False,
            "agent_id": agent_id,
            "status": "running",
            "error": "Stop the subagent before deleting it",
        }
    if record.get("completion_pending"):
        return {
            "ok": False,
            "agent_id": agent_id,
            "status": "completion_pending",
            "error": "Wait for the parent completion notification",
        }
    if not await delete_subagent_activity(record):
        return {
            "ok": False,
            "agent_id": agent_id,
            "status": "delete_failed",
            "error": "Subagent activity surface could not be deleted",
        }
    session.subagent_records.pop(agent_id, None)
    state["interrupted_subagents"] = [
        item
        for item in state.get("interrupted_subagents", [])
        if str(item.get("id")) != agent_id
    ]
    save_state()
    return {"ok": True, "agent_id": agent_id, "status": "deleted"}


async def subagent_tool(
    args: dict[str, Any],
    *,
    chat_id: ConversationId | None,
    execution_context: dict[str, Any],
) -> dict[str, Any]:
    task = str(
        args.get(
            "task",
            "",
        )
    ).strip()
    requested_id = str(
        args.get(
            "agent_id",
            "",
        )
    ).strip()
    requested_provider = str(args.get("model_provider") or "").strip()
    requested_model = str(args.get("model") or "").strip()
    delete_requested = bool(args.get("delete", False))
    if not task and (requested_provider or requested_model):
        return {
            "ok": False,
            "error": "model_provider and model are only valid when starting a subagent",
        }
    if delete_requested:
        if task:
            return {
                "ok": False,
                "error": "delete cannot be combined with task",
            }
        if not requested_id:
            return {
                "ok": False,
                "error": "delete requires agent_id",
            }
        if chat_id is None or (
            session.conversation_key(chat_id) != session.current_runtime().key
        ):
            return {
                "ok": False,
                "error": "subagent conversation does not match the current chat",
            }
        return await delete_subagent_record(requested_id)
    if not task:
        if requested_id:
            record = (
                session.subagent_records.get(
                    requested_id
                )
            )
            if not record:
                return {
                    "ok": False,
                    "error": (
                        f"subagent {requested_id} not found"
                    ),
                }
            return {
                "ok": True,
                "subagent": subagent_status(record),
            }
        return {
            "ok": True,
            "subagents":
                subagent_summaries(),
        }
    if requested_id and (requested_provider or requested_model):
        return {
            "ok": False,
            "error": (
                "A continued subagent keeps its original model target; "
                "start a new subagent to use another provider or model"
            ),
        }
    if len(task) > MAX_TOOL_OUTPUT:
        return {
            "ok": False,
            "error": (
                "subagent task exceeds "
                f"{MAX_TOOL_OUTPUT} characters"
            ),
        }
    if chat_id is None:
        return {
            "ok": False,
            "error": (
                "subagent requires a user-facing parent agent"
            ),
        }
    if session.conversation_key(chat_id) != session.current_runtime().key:
        return {
            "ok": False,
            "error": "subagent conversation does not match the current chat",
        }
    parent_depth = int(
        execution_context.get(
            "depth",
            0,
        )
    )
    immediate_parent_session_id = str(
        execution_context.get("session_id") or ""
    ) or None
    root_session_id = str(
        execution_context.get("root_session_id")
        or immediate_parent_session_id
        or ""
    ) or None
    selected_target: ModelTarget | None = None
    if not requested_id:
        raw_parent_target = execution_context.get("model_target")
        try:
            parent_target = (
                ModelTarget.from_value(raw_parent_target)
                if raw_parent_target is not None
                else session.current_model_target()
            )
            selected_target = MODEL_CONFIGURATION.select_target(
                parent=parent_target,
                provider_id=requested_provider or None,
                model=requested_model or None,
            )
            # Resolve credentials before admitting durable work.
            model_provider(selected_target).api_key()
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
    if (
        parent_depth > 0
        and not execution_context.get(
            "subagents_allowed",
            False,
        )
    ):
        return {
            "ok": False,
            "error": (
                "Nested subagents were not authorized"
            ),
        }
    depth = parent_depth + 1
    if depth > MAX_SUBAGENT_DEPTH:
        return {
            "ok": False,
            "error": (
                "Maximum subagent depth reached"
            ),
        }
    background = bool(
        args.get(
            "background",
            False,
        )
    )
    if (
        parent_depth > 0
        and background
    ):
        return {
            "ok": False,
            "error": (
                "Nested subagents must run in the foreground"
            ),
        }
    previous_completion: dict[str, Any] | None = None
    if requested_id:
        record = (
            session.subagent_records.get(
                requested_id
            )
        )
        if not record:
            return {
                "ok": False,
                "error": (
                    f"subagent {requested_id} not found"
                ),
            }
        try:
            # Continuations are bound to the child's durable, immutable target.
            # The caller's current target is irrelevant after child creation.
            model_provider(record_model_target(record)).api_key()
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        owner_task = session.subagent_tasks.get(requested_id)
        if owner_task is not None and not owner_task.done():
            if record.get("status") != "running":
                return {
                    "ok": False,
                    "agent_id": requested_id,
                    "status": "finalizing",
                    "error": "Subagent is finalizing its current run",
                }
        if record.get("status") == "running":
            pending_inputs = record.setdefault(
                "pending_inputs",
                [],
            )
            if len(pending_inputs) >= MAX_SUBAGENT_PENDING_INPUTS:
                return {
                    "ok": False,
                    "agent_id": requested_id,
                    "status": "queue_full",
                    "error": (
                        "Subagent pending-input queue is full "
                        f"({MAX_SUBAGENT_PENDING_INPUTS})"
                    ),
                }
            record[
                "last_task"
            ] = task
            record["task_count"] = int(
                record.get("task_count") or 1
            ) + 1
            pending_inputs.append(task)
            save_state()
            await append_subagent_activity_input(record, task)
            log_event(
                "subagent",
                "steering_queued",
                {
                    "agent_id":
                        requested_id,
                    "input_chars": len(task),
                },
            )
            return {
                "ok": True,
                "agent_id":
                    requested_id,
                "status":
                    "steering_queued",
            }
        if record.get("completion_pending"):
            previous_completion = subagent_status(record)
        allow_subagents = (
            bool(
                args[
                    "allow_subagents"
                ]
            )
            if "allow_subagents"
            in args
            else bool(
                record.get(
                    "allow_subagents",
                    False,
                )
            )
        )
        await finish_subagent_activity(record)
        if background:
            capacity_error = _background_capacity_error()
            if capacity_error is not None:
                return capacity_error
        continue_record(
            record,
            task,
            chat_id=chat_id,
            depth=depth,
            background=background,
            allow_subagents=(
                allow_subagents
                and depth
                < MAX_SUBAGENT_DEPTH
            ),
        )
    else:
        if selected_target is None:
            return {"ok": False, "error": "model target was not selected"}
        open_surfaces = sum(
            bool(item.get("activity_surface_id"))
            for item in session.subagent_records.values()
        )
        if background and open_surfaces >= MAX_SUBAGENT_RECORDS:
            return {
                "ok": False,
                "status": "capacity_reached",
                "error": (
                    "Persistent subagent topic capacity reached "
                    f"({MAX_SUBAGENT_RECORDS})"
                ),
            }
        if background:
            capacity_error = _background_capacity_error()
            if capacity_error is not None:
                return capacity_error
        allow_subagents = bool(
            args.get(
                "allow_subagents",
                False,
            )
        )
        record = new_record(
            task,
            target=selected_target,
            chat_id=chat_id,
            depth=depth,
            background=background,
            allow_subagents=(
                allow_subagents
                and depth
                < MAX_SUBAGENT_DEPTH
            ),
            parent_session_id=immediate_parent_session_id,
            root_session_id=root_session_id,
        )
    agent_id = str(
        record["id"]
    )
    session.subagent_records[
        agent_id
    ] = record
    prune_subagent_records()
    save_state()
    log_event(
        "subagent",
        "started",
        lifecycle_record(
            record
        ),
    )
    generation = max(1, int(record.get("run_generation") or 1))
    if not background and record.get("activity_surface_id"):
        await open_subagent_activity(record)
    if background:
        global _background_reservations
        _background_reservations += 1
        try:
            await open_subagent_activity(record)
            task_handle = asyncio.create_task(
                run_background_subagent(
                    record
                )
            )
            session.subagent_tasks[
                agent_id
            ] = task_handle
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["error"] = "Subagent start was cancelled"
            record["completed_at"] = int(time.time())
            save_state()
            await finish_subagent_activity(record)
            raise
        finally:
            _background_reservations -= 1
        return {
            "ok": True,
            "agent_id": agent_id,
            "status": "running",
            "completion_notification": "automatic",
            **(
                {"previous_completion": previous_completion}
                if previous_completion is not None
                else {}
            ),
        }
    try:
        result = await run_subagent(
            agent_id=agent_id,
            chat_id=chat_id,
            depth=depth,
            allow_subagents=bool(
                record[
                    "allow_subagents"
                ]
            ),
        )
        record["result"] = _bounded_record_text(result)
        record["status"] = "completed"
        save_state()
        return {
            "ok": True,
            "agent_id": agent_id,
            "status": "completed",
            "result": record["result"],
            **(
                {"previous_completion": previous_completion}
                if previous_completion is not None
                else {}
            ),
        }
    except asyncio.CancelledError:
        record["status"] = "cancelled"
        record["error"] = (
            "Subagent was cancelled"
        )
        record["completed_at"] = int(
            time.time()
        )
        save_interrupted_subagent(
            record
        )
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = _bounded_record_text(
            f"{type(exc).__name__}: {exc}"
        )
        save_state()
        return {
            "ok": False,
            "agent_id": agent_id,
            "status": "failed",
            "error": record["error"],
            **(
                {"previous_completion": previous_completion}
                if previous_completion is not None
                else {}
            ),
        }
    finally:
        record["completed_at"] = int(
            time.time()
        )
        await finish_subagent_activity(
            record,
            expected_generation=generation,
        )
        prune_subagent_records()
        save_state()
        log_event(
            "subagent",
            "completed",
            lifecycle_record(
                record
            ),
        )


def completion_prompt(
    record: dict[str, Any],
) -> str:
    payload = {
        "agent_id": record.get("id"),
        "task": (
            record.get("last_task")
            or record.get("task")
        ),
        "status": record.get("status"),
        "result": record.get("result"),
        "error": record.get("error"),
    }
    return (
        "A background subagent has finished. Assess its result as the parent "
        "agent and send the user the relevant outcome exactly once. This is "
        "the authoritative completion for this run; do not poll the subagent "
        "or start a wait command. Do not describe this notification as a new "
        "user request.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


def pending_subagent_completions() -> int:
    return sum(
        bool(record.get("completion_pending"))
        for record in session.subagent_records.values()
    )


async def requeue_deferred_subagent_completions() -> int:
    """Retry deferred completion delivery after explicit user activity."""
    records = [
        record
        for record in session.subagent_records.values()
        if record.get("completion_pending")
        and int(record.get("completion_attempts", 0)) > 0
    ]
    for record in records:
        record["completion_attempts"] = 0
        _queue_subagent_completion(record)
    if records:
        save_state()
    return len(records)


def _queue_subagent_completion(record: dict[str, Any]) -> bool:
    from controller import enqueue_runtime_event
    from runtime_events import RuntimeEventKind

    chat_id = record.get("chat_id")
    if chat_id is None:
        return False
    owner_key = str(record.get("chat_key") or "")
    if owner_key != session.conversation_key(chat_id):
        log_event(
            "subagent",
            "completion_owner_mismatch",
            {"agent_id": record.get("id")},
        )
        return False
    agent_id = str(record.get("id") or "")
    generation = max(1, int(record.get("run_generation") or 1))
    with session.bound_chat(chat_id):
        live_record = session.subagent_records.get(agent_id)
        if (
            not live_record
            or not live_record.get("completion_pending")
        ):
            return False
        live_generation = max(
            1,
            int(live_record.get("run_generation") or 1),
        )
        if live_generation != generation:
            return False
        try:
            completion = enqueue_runtime_event(
                chat_id,
                RuntimeEventKind.SUBAGENT_COMPLETED,
                completion_prompt(record),
                is_active=lambda current=live_record, expected=generation: bool(
                    current.get("completion_pending")
                    and max(1, int(current.get("run_generation") or 1))
                    == expected
                ),
            )
        except Exception as exc:
            log_event(
                "subagent",
                "completion_delivery_error",
                {
                    "agent_id": record.get("id"),
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            live_record["completion_attempts"] = 1
            save_state()
            return False

    def completed(done: asyncio.Future[bool]) -> None:
        try:
            delivered = not done.cancelled() and bool(done.result())
        except Exception as exc:
            delivered = False
            log_event(
                "subagent",
                "completion_delivery_error",
                {
                    "agent_id": agent_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        with session.bound_chat(chat_id):
            current = session.subagent_records.get(agent_id)
            if (
                current is None
                or not current.get("completion_pending")
                or max(1, int(current.get("run_generation") or 1))
                != generation
            ):
                return
            if delivered:
                current["completion_pending"] = False
                current["completion_notified_at"] = int(time.time())
                current["completion_attempts"] = 0
                prune_subagent_records()
                event = "completion_delivered"
                payload = {"agent_id": agent_id}
            else:
                attempts = int(current.get("completion_attempts") or 0) + 1
                current["completion_attempts"] = attempts
                event = "completion_deferred"
                payload = {"agent_id": agent_id, "attempts": attempts}
            save_state()
            log_event("subagent", event, payload)

    completion.add_done_callback(completed)
    return True


def restore_pending_subagent_completions() -> int:
    """Queue persisted completions once when a chat runtime starts."""
    records = [
        record
        for record in session.subagent_records.values()
        if record.get("completion_pending")
    ]
    for record in records:
        record["completion_attempts"] = 0
        _queue_subagent_completion(record)
    if records:
        save_state()
    return len(records)


async def cancel_background_subagents(*, discard_pending: bool = False) -> int:
    tasks: list[asyncio.Task] = []
    for agent_id, task in list(
        session.subagent_tasks.items()
    ):
        if task.done():
            continue
        record = session.subagent_records.get(
            agent_id
        )
        if record:
            record[
                "notify_completion"
            ] = False
        tasks.append(
            task
        )
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
    if discard_pending:
        changed = False
        for record in session.subagent_records.values():
            if record.get("completion_pending"):
                record["completion_pending"] = False
                record["completion_attempts"] = 0
                changed = True
        if changed:
            save_state()
    return len(tasks)


async def cancel_background_subagent(agent_id: str) -> dict[str, Any]:
    """Cancel one live background subagent without notifying its parent."""
    record = session.subagent_records.get(agent_id)
    if record is None:
        return {
            "ok": False,
            "kind": "subagent",
            "id": agent_id,
            "error": f"unknown subagent: {agent_id}",
        }
    task = session.subagent_tasks.get(agent_id)
    if task is None or task.done() or record.get("status") != "running":
        return {
            "ok": False,
            "kind": "subagent",
            "id": agent_id,
            "status": record.get("status"),
            "error": "subagent is not running in the background",
        }

    record["notify_completion"] = False
    record["completion_pending"] = False
    record["completion_attempts"] = 0
    save_state()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    cancelled = record.get("status") == "cancelled"
    return {
        "ok": cancelled,
        "kind": "subagent",
        "id": agent_id,
        "status": record.get("status"),
        **(
            {}
            if cancelled
            else {"error": "subagent finished before cancellation completed"}
        ),
    }
