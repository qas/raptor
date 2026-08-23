"""OpenAI-compatible Responses API client."""
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from chat_provider import ConversationId

from config import (
    BASE_INSTRUCTIONS,
    CHAT_STREAM_INTERVAL,
    CHAT_STREAMING,
    TOOLS,
    RESPONSES_API_KEY,
    RESPONSES_BASE_URL,
    RESPONSES_MODEL,
    RESPONSES_REASONING_EFFORT,
    RESPONSES_REASONING_SUMMARY,
    RESPONSES_MAX_RETRIES,
    RESPONSES_RETRY_BASE_SECONDS,
)
from session import save_state, state
import session
from chat_runtime import send_draft, send_reasoning_summary
from observability import log_event
from engine import estimate_tokens
from response_errors import (
    ContextLengthError,
    IncompleteResponsesStreamError,
    TransientResponsesError,
)


_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_ResponseT = TypeVar("_ResponseT")


def is_transient_responses_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (IncompleteResponsesStreamError, httpx.TransportError),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP_STATUSES
    return False


async def retry_transient_response(
    request: Callable[[], Awaitable[_ResponseT]],
    *,
    operation: str,
    log_data: dict[str, Any] | None = None,
    max_retries: int | None = None,
    retry_base_seconds: float | None = None,
) -> _ResponseT:
    max_retries = (
        RESPONSES_MAX_RETRIES
        if max_retries is None
        else max(0, max_retries)
    )
    retry_base_seconds = (
        RESPONSES_RETRY_BASE_SECONDS
        if retry_base_seconds is None
        else max(0.0, retry_base_seconds)
    )
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return await request()
        except ContextLengthError:
            raise
        except Exception as exc:
            if not is_transient_responses_error(exc):
                raise
            payload = {
                "operation": operation,
                "attempt": attempt,
                "max_retries": max_retries,
                "type": type(exc).__name__,
                "message": str(exc),
                **(log_data or {}),
            }
            if attempt >= total_attempts:
                log_event("responses", "retry_exhausted", payload)
                raise TransientResponsesError(
                    f"{operation} failed after {max_retries} retries: {exc}"
                ) from exc
            delay = retry_base_seconds * (2 ** (attempt - 1))
            payload["delay_seconds"] = delay
            log_event("responses", "retrying", payload)
            if delay:
                await asyncio.sleep(delay)
    raise AssertionError("unreachable response retry state")


def parse_context_length_error(
    response: httpx.Response,
) -> ContextLengthError | None:
    if response.status_code not in {400, 413, 500}:
        return None

    try:
        body = response.json()
    except Exception:
        body = None

    error = (
        body.get("error", {})
        if isinstance(body, dict)
        else {}
    )
    if not isinstance(error, dict):
        error = {}
    # Some backends put overflow fields on the root object.
    root = body if isinstance(body, dict) else {}

    message = str(
        error.get("message")
        or root.get("message")
        or response.text
        or ""
    )

    error_type = str(
        error.get("type")
        or error.get("code")
        or root.get("type")
        or root.get("code")
        or ""
    ).lower()

    message_lower = message.lower()
    combined = f"{error_type} {message_lower}"

    looks_like_context_error = (
        "context" in error_type
        or "exceed_context_size" in combined
        or "context size has been exceeded" in message_lower
        or "context size" in message_lower
        or "context length" in message_lower
        or "maximum context" in message_lower
        or "exceeds the available context" in message_lower
    )

    # 500 is only recoverable when the payload clearly indicates overflow.
    if response.status_code == 500 and not looks_like_context_error:
        return None
    if not looks_like_context_error:
        return None

    prompt_tokens = (
        error.get("n_prompt_tokens")
        or error.get("prompt_tokens")
        or root.get("n_prompt_tokens")
        or root.get("prompt_tokens")
    )
    context_tokens = (
        error.get("n_ctx")
        or error.get("context_tokens")
        or root.get("n_ctx")
        or root.get("context_tokens")
    )

    try:
        prompt_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        prompt_tokens = None

    try:
        context_tokens = int(context_tokens)
    except (TypeError, ValueError):
        context_tokens = None

    return ContextLengthError(
        message or "request exceeded model context",
        prompt_tokens=prompt_tokens,
        context_tokens=context_tokens,
    )


def auth_headers() -> dict[str, str]:
    if RESPONSES_API_KEY:
        return {
            "Authorization":
                f"Bearer {RESPONSES_API_KEY}"
        }

    return {}


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

async def _list_models_once() -> list[str]:
    response = await session.responses.get(
        f"{RESPONSES_BASE_URL}/models",
        headers=auth_headers(),
    )

    response.raise_for_status()
    return [
        model["id"]
        for model
        in response.json().get(
            "data",
            [],
        )
        if model.get(
            "id"
        )
    ]


async def list_models() -> list[str]:
    return await retry_transient_response(
        _list_models_once,
        operation="Responses model listing",
    )


async def ensure_model() -> str:
    model = state.get(
        "model"
    )

    if model:
        return str(
            model
        )

    models = await list_models()

    if not models:
        raise RuntimeError(
            "Responses API returned no models"
        )

    state["model"] = (
        models[0]
    )

    save_state()

    return models[0]


def instructions(
    extra: str = "",
) -> str:
    parts = [
        BASE_INSTRUCTIONS
    ]
    if extra:
        parts.append(
            extra
        )
    return "\n\n".join(
        parts
    )


def build_response_payload(
    input_items: list[
        dict[str, Any]
    ],
    *,
    tools: list[
        dict[str, Any]
    ]
    | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int
    | None = None,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": input_items,
        "instructions": instructions(
            extra_instructions
        ),
        "stream": stream,
    }

    if tools:
        payload[
            "tools"
        ] = tools

        payload[
            "parallel_tool_calls"
        ] = False

    if max_output_tokens is not None:
        payload[
            "max_output_tokens"
        ] = max_output_tokens

    if reasoning_effort is not None or reasoning_summary is not None:
        payload["reasoning"] = {}
        if reasoning_effort is not None:
            payload["reasoning"]["effort"] = reasoning_effort
        if reasoning_summary is not None:
            payload["reasoning"]["summary"] = reasoning_summary

    return payload


async def _responses_create_once(
    input_items: list[
        dict[str, Any]
    ],
    *,
    tools: list[
        dict[str, Any]
    ]
    | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int
    | None = None,
    reasoning_effort: str | None = RESPONSES_REASONING_EFFORT,
) -> dict[str, Any]:
    payload = build_response_payload(
        input_items,
        tools=tools,
        extra_instructions=(
            extra_instructions
        ),
        max_output_tokens=(
            max_output_tokens
        ),
        reasoning_effort=reasoning_effort,
        stream=False,
    )

    payload[
        "model"
    ] = await ensure_model()
    response = await session.responses.post(
        f"{RESPONSES_BASE_URL}/responses",
        headers=auth_headers(),
        json=payload,
        timeout=None,
    )

    if response.is_error:
        context_error = parse_context_length_error(
            response
        )
        if context_error:
            raise context_error

    response.raise_for_status()
    return response.json()


async def responses_create(
    input_items: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = RESPONSES_REASONING_EFFORT,
) -> dict[str, Any]:
    async def request() -> dict[str, Any]:
        return await _responses_create_once(
            input_items,
            tools=tools,
            extra_instructions=extra_instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    return await retry_transient_response(
        request,
        operation="Responses request",
    )


def build_stateless_response_payload(
    input_items: list[dict[str, Any]],
    model: str,
    *,
    tools: list[dict[str, Any]] | None = TOOLS,
) -> dict[str, Any]:
    """Build an instruction-free request for the in-memory ``/ask`` loop."""
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["parallel_tool_calls"] = False
    if RESPONSES_REASONING_EFFORT is not None:
        payload["reasoning"] = {
            "effort": RESPONSES_REASONING_EFFORT,
        }
    return payload


async def _stateless_response_once(
    input_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call the selected model without instructions or durable history."""
    model = str(state.get("model") or RESPONSES_MODEL or "")
    if not model:
        models = await list_models()
        if not models:
            raise RuntimeError("Responses API returned no models")
        model = models[0]
    response = await session.responses.post(
        f"{RESPONSES_BASE_URL}/responses",
        headers=auth_headers(),
        json=build_stateless_response_payload(input_items, model),
        timeout=None,
    )
    if response.is_error:
        context_error = parse_context_length_error(response)
        if context_error:
            raise context_error
    response.raise_for_status()
    return response.json()


async def stateless_response(
    input_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return await retry_transient_response(
        lambda: _stateless_response_once(input_items),
        operation="Stateless Responses request",
    )


async def _responses_create_stream_once(
    chat_id: ConversationId,
    input_items: list[
        dict[str, Any]
    ],
    *,
    tools: list[
        dict[str, Any]
    ]
    | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int
    | None = None,
    reasoning_effort: str | None = RESPONSES_REASONING_EFFORT,
    reasoning_summary: str | None = RESPONSES_REASONING_SUMMARY,
) -> dict[str, Any]:
    payload = build_response_payload(
        input_items,
        tools=tools,
        extra_instructions=(
            extra_instructions
        ),
        max_output_tokens=(
            max_output_tokens
        ),
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        stream=True,
    )

    payload[
        "model"
    ] = await ensure_model()
    draft_id = max(
        1,
        int(
            time.time_ns()
            & 0x7FFFFFFF
        ),
    )

    text_buffer = ""
    last_draft = 0.0
    last_draft_text = ""
    draft_task: asyncio.Task[None] | None = None

    async def publish_draft(snapshot: str) -> None:
        try:
            await send_draft(chat_id, draft_id, snapshot)
        except Exception as exc:
            log_event(
                "responses",
                "draft_error",
                {
                    "conversation_id": str(chat_id),
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    async def publish_reasoning_summary(delta: str) -> None:
        try:
            await send_reasoning_summary(chat_id, delta)
        except Exception as exc:
            log_event(
                "responses",
                "reasoning_summary_error",
                {
                    "conversation_id": str(chat_id),
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    final_response: (
        dict[str, Any]
        | None
    ) = None

    async with session.responses.stream(
        "POST",
        f"{RESPONSES_BASE_URL}/responses",
        headers=auth_headers(),
        json=payload,
        timeout=None,
    ) as response:
        if response.is_error:
            await response.aread()
            log_event(
                "responses",
                "http_error",
                {
                    "status":
                        response.status_code,
                    "body_chars": len(response.text),
                },
            )

            context_error = parse_context_length_error(
                response
            )
            if context_error:
                raise context_error

        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith(
                "data:"
            ):
                continue

            raw = line[
                5:
            ].strip()

            if (
                not raw
                or raw == "[DONE]"
            ):
                continue

            try:
                event = json.loads(
                    raw
                )

            except json.JSONDecodeError:
                continue

            event_type = event.get(
                "type"
            )
            summary_delta = reasoning_summary_delta(event)
            if summary_delta:
                await publish_reasoning_summary(summary_delta)
            if (
                event_type
                == "response.output_text.delta"
            ):
                text_buffer += str(
                    event.get(
                        "delta"
                    )
                    or ""
                )

                now = (
                    time.monotonic()
                )

                if (
                    CHAT_STREAMING
                    and (
                        now
                        - last_draft
                        >= CHAT_STREAM_INTERVAL
                    )
                    and (
                        draft_task is None
                        or draft_task.done()
                    )
                ):
                    last_draft_text = text_buffer
                    draft_task = asyncio.create_task(
                        publish_draft(last_draft_text)
                    )
                    last_draft = now

            elif (
                event_type
                == "response.completed"
            ):
                completed = event.get(
                    "response"
                )

                if isinstance(
                    completed,
                    dict,
                ):
                    final_response = (
                        completed
                    )

            elif event_type in {
                "response.failed",
                "error",
            }:
                raise RuntimeError(
                    event.get(
                        "message"
                    )
                    or event.get(
                        "error"
                    )
                    or str(event)
                )

    if (
        CHAT_STREAMING
        and text_buffer
    ):
        if draft_task is not None:
            await draft_task
        if text_buffer != last_draft_text:
            await publish_draft(text_buffer)

    if final_response is None:
        raise IncompleteResponsesStreamError(
            "Responses stream ended without "
            "response.completed"
        )

    return final_response


async def responses_create_stream(
    chat_id: ConversationId,
    input_items: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = RESPONSES_REASONING_EFFORT,
    reasoning_summary: str | None = RESPONSES_REASONING_SUMMARY,
) -> dict[str, Any]:
    async def request() -> dict[str, Any]:
        return await _responses_create_stream_once(
            chat_id,
            input_items,
            tools=tools,
            extra_instructions=extra_instructions,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )

    return await retry_transient_response(
        request,
        operation="Responses stream",
        log_data={"conversation_id": str(chat_id)},
    )


def reasoning_summary_delta(event: dict[str, Any]) -> str:
    """Return only public reasoning-summary text, never raw reasoning."""
    if event.get("type") != "response.reasoning_summary_text.delta":
        return ""
    return str(event.get("delta") or "")


def estimate_response_request_tokens(
    input_items: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = TOOLS,
    extra_instructions: str = "",
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> int:
    payload = build_response_payload(
        input_items,
        tools=tools,
        extra_instructions=extra_instructions,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        stream=False,
    )

    return estimate_tokens(payload)
