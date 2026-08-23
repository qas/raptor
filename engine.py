"""Shared agent execution utilities."""
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from config import MAX_TOOL_OUTPUT
from observability import log_event, tool_activity


def interrupted_tool_result() -> dict[str, Any]:
    """Return the durable result paired with an interrupted tool call."""
    return {
        "ok": False,
        "status": "interrupted",
        "error": (
            "Tool execution was interrupted by the user. It may have "
            "partially changed external state; inspect before retrying."
        ),
    }


def function_call_output(
    call: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    output = json.dumps(result, ensure_ascii=False)
    if len(output) > MAX_TOOL_OUTPUT:
        output = _bounded_result_envelope(result, output)
    return {
        "type": "function_call_output",
        "call_id": call["call_id"],
        "output": output,
    }


def _bounded_result_envelope(
    result: dict[str, Any],
    encoded: str,
) -> str:
    """Keep oversized tool output valid JSON and within the context cap."""
    marker = "\n... [tool result truncated] ...\n"

    def candidate(retained: int) -> str:
        head = retained // 2
        tail = retained - head
        preview = (
            encoded[:head]
            + marker
            + (encoded[-tail:] if tail else "")
        )
        return json.dumps(
            {
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "truncated": True,
                "original_chars": len(encoded),
                "preview": preview,
            },
            ensure_ascii=False,
        )

    low = 0
    high = len(encoded)
    best = candidate(0)
    while low <= high:
        retained = (low + high) // 2
        attempt = candidate(retained)
        if len(attempt) <= MAX_TOOL_OUTPUT:
            best = attempt
            low = retained + 1
        else:
            high = retained - 1
    return best

CreateResponse = Callable[
    [list[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]
ExecuteCall = Callable[
    [dict[str, Any]],
    Awaitable[dict[str, Any]],
]
DrainInputs = Callable[
    [
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    Awaitable[int],
]
Checkpoint = Callable[
    [list[dict[str, Any]]],
    None,
]
CompactContext = Callable[
    [list[dict[str, Any]]],
    Awaitable[list[dict[str, Any]] | None],
]
RecordItems = Callable[
    [list[dict[str, Any]], str],
    None,
]


def estimate_tokens(value: Any) -> int:
    text = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    # Intentionally conservative approximation.
    # 3 chars/token instead of 4 prevents systematic undercounting.
    return max(1, (len(text) + 2) // 3)


def response_output(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    output = response.get("output", [])
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict)
    ]


def response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response_output(response):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if (
                content.get("type") in {"output_text", "text"}
                and content.get("text")
            ):
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def response_calls(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        item
        for item in response_output(response)
        if item.get("type") == "function_call"
    ]


async def run_agent(
    *,
    work: list[dict[str, Any]],
    create_response: CreateResponse,
    execute_call: ExecuteCall,
    source: str,
    max_tool_rounds: int,
    agent_id: str | None = None,
    drain_inputs: DrainInputs | None = None,
    checkpoint: Checkpoint | None = None,
    compact_context: CompactContext | None = None,
    record_items: RecordItems | None = None,
) -> dict[str, Any]:
    turn_inputs: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    tool_rounds = 0
    context_compacted = False
    metadata = {"agent_id": agent_id} if agent_id else {}
    log_event(source, "turn_started", {**metadata, "work_len": len(work)})
    while True:
        if drain_inputs:
            await drain_inputs(work, turn_inputs)
        if compact_context:
            compacted = await compact_context(work)
            if compacted is not None:
                work[:] = compacted
                context_compacted = True
        log_event(source, "thinking", metadata)
        response = await create_response(work)
        log_event(
            source,
            "response",
            {
                **metadata,
                "id": response.get("id"),
                "status": response.get("status"),
                "output_types": [
                    str(item.get("type") or "unknown")
                    for item in response_output(response)
                ],
                "usage": response.get("usage"),
            },
        )
        if drain_inputs:
            applied = await drain_inputs(work, turn_inputs)
        else:
            applied = 0
        if applied:
            log_event(source, "replanning", metadata)
            continue
        calls = response_calls(response)
        if not calls:
            text = response_text(response)
            if not text:
                raise RuntimeError("Agent returned no text")
            final_output = response_output(response)
            if record_items and final_output:
                record_items(final_output, "assistant")
            result = {
                "text": text,
                "output": final_output,
                "inputs": turn_inputs,
                "context_compacted": context_compacted,
                "tool_events": tool_events,
            }
            log_event(
                source,
                "turn_completed",
                {**metadata, "output_chars": len(text)},
            )
            return result
        if max_tool_rounds and tool_rounds >= max_tool_rounds:
            raise RuntimeError(
                "Configured tool-round limit reached"
            )
        tool_rounds += 1
        accepted_output = response_output(response)
        if record_items and accepted_output:
            record_items(accepted_output, "assistant")
        work.extend(accepted_output)
        for call in calls:
            event = {
                "call": call,
                "status": "in_progress",
                "result": None,
            }
            tool_events.append(event)
            if checkpoint:
                checkpoint(tool_events)
            log_event(
                source,
                "tool_call",
                {
                    **metadata,
                    "activity": tool_activity(call),
                    "name": call.get("name"),
                    "call_id": call.get("call_id"),
                },
            )
            try:
                call_result = await execute_call(call)
            except asyncio.CancelledError:
                call_result = interrupted_tool_result()
                event["status"] = "interrupted"
                event["result"] = call_result
                if checkpoint:
                    checkpoint(tool_events)
                output_item = function_call_output(call, call_result)
                if record_items:
                    record_items([output_item], "tool")
                work.append(output_item)
                log_event(
                    source,
                    "tool_interrupted",
                    {
                        **metadata,
                        "name": call.get("name"),
                        "call_id": call.get("call_id"),
                    },
                )
                raise
            except Exception as exc:
                call_result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if not isinstance(call_result, dict):
                call_result = {
                    "ok": False,
                    "error": "Tool returned a non-object result",
                }
            event["status"] = (
                "completed" if call_result.get("ok") else "failed"
            )
            event["result"] = call_result
            if checkpoint:
                checkpoint(tool_events)
            log_event(
                source,
                "tool_result",
                {
                    **metadata,
                    "call_id": call.get("call_id"),
                    "ok": call_result.get("ok"),
                    "status": call_result.get("status"),
                    "has_error": bool(call_result.get("error")),
                },
            )
            output_item = function_call_output(call, call_result)
            if record_items:
                record_items([output_item], "tool")
            work.append(output_item)
