"""OpenAI-compatible Responses API client."""
import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
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
    MalformedToolCallError,
    PartialResponsesStreamError,
    TransientResponsesError,
)


_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_ResponseT = TypeVar("_ResponseT")
MODEL_LIST_TIMEOUT_SECONDS = 10.0


@dataclass
class ResponsesStreamReplayGuard:
    """Prevent replay after a stream has projected public output."""

    public_output_seen: bool = False

    def observe(self, snapshot: str) -> None:
        if snapshot:
            self.public_output_seen = True

    def wrap(
        self,
        callback: Callable[[str], Awaitable[None]] | None,
    ) -> Callable[[str], Awaitable[None]] | None:
        if callback is None:
            return None

        async def project(snapshot: str) -> None:
            self.observe(snapshot)
            await callback(snapshot)

        return project

    def reject_unsafe_replay(
        self,
        exc: Exception,
        *,
        operation: str,
    ) -> None:
        if self.public_output_seen and is_transient_responses_error(exc):
            raise PartialResponsesStreamError(
                f"{operation} interrupted after public output; "
                "automatic replay was suppressed"
            ) from exc


def _error_text(payload: Any) -> str:
    """Return a bounded searchable rendering of a provider error."""
    parts: list[str] = []
    character_count = 0

    def collect(value: Any, depth: int = 0) -> None:
        nonlocal character_count
        if character_count >= 4096 or depth > 4:
            return
        if isinstance(value, dict):
            for key in ("type", "code", "message", "error", "response"):
                if key in value:
                    collect(value[key], depth + 1)
            return
        if isinstance(value, list):
            for item in value:
                collect(item, depth + 1)
            return
        text = str(value)
        parts.append(text)
        character_count += len(text)

    collect(payload)
    return " ".join(parts)[:4096]


def parse_malformed_tool_call_error(
    payload: Any,
) -> MalformedToolCallError | None:
    """Recognize rejection of invalid model-generated call arguments."""
    text = _error_text(payload).lower()
    mentions_call = any(
        marker in text
        for marker in (
            "tool call",
            "tool_call",
            "function call",
            "function_call",
        )
    )
    invalid_json = "json" in text and any(
        marker in text
        for marker in (
            "invalid",
            "malformed",
            "parse",
            "syntax",
            "unterminated",
        )
    )
    if not (mentions_call and "argument" in text and invalid_json):
        return None
    return MalformedToolCallError(
        "provider rejected malformed model-generated tool arguments"
    )


def parse_http_response_error(
    response: httpx.Response,
) -> RuntimeError | None:
    """Classify provider failures that require agent-level handling."""
    context_error = parse_context_length_error(response)
    if context_error:
        return context_error
    if not response.is_error:
        return None
    try:
        payload: Any = response.json()
    except ValueError:
        payload = response.text
    return parse_malformed_tool_call_error(payload)


def is_transient_responses_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (IncompleteResponsesStreamError, httpx.TransportError),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP_STATUSES
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return a valid HTTP Retry-After delay from a failed response."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            delay = parsedate_to_datetime(raw).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)


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
            retry_after = retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
                payload["retry_after_seconds"] = retry_after
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
    except ValueError:
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
        timeout=httpx.Timeout(MODEL_LIST_TIMEOUT_SECONDS),
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


async def list_models(*, max_retries: int | None = None) -> list[str]:
    return await retry_transient_response(
        _list_models_once,
        operation="Responses model listing",
        max_retries=max_retries,
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


def validate_chronological_input(input_items: list[dict[str, Any]]) -> None:
    """Require all instruction authority to use the top-level field."""
    for item in input_items:
        if item.get("role") in {"developer", "system"}:
            raise ValueError(
                "Chronological input cannot contain developer or system "
                "messages; use the instructions field"
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
    validate_chronological_input(input_items)
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
        classified = parse_http_response_error(response)
        if classified:
            raise classified

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
    validate_chronological_input(input_items)
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
        classified = parse_http_response_error(response)
        if classified:
            raise classified
    response.raise_for_status()
    return response.json()


async def stateless_response(
    input_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return await retry_transient_response(
        lambda: _stateless_response_once(input_items),
        operation="Stateless Responses request",
    )


async def stream_response_payload(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    on_text: Callable[[str], Awaitable[None]] | None = None,
    on_reasoning_summary: Callable[[str], Awaitable[None]] | None = None,
    log_source: str = "responses",
    log_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream one Responses request and expose cumulative public output."""
    final_response: dict[str, Any] | None = None
    text = ""
    reasoning_summary = ""
    async with session.responses.stream(
        "POST",
        url,
        headers=headers,
        json=payload,
        timeout=None,
    ) as response:
        if response.is_error:
            await response.aread()
            log_event(
                log_source,
                "http_error",
                {
                    **(log_data or {}),
                    "status": response.status_code,
                    "body_chars": len(response.text),
                },
            )
            classified = parse_http_response_error(response)
            if classified:
                raise classified
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            summary_delta = reasoning_summary_delta(event)
            if summary_delta and on_reasoning_summary is not None:
                reasoning_summary += summary_delta
                await on_reasoning_summary(reasoning_summary)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                text += str(event.get("delta") or "")
                if on_text is not None:
                    await on_text(text)
            elif event_type == "response.completed":
                completed = event.get("response")
                if isinstance(completed, dict):
                    final_response = completed
            elif event_type in {"response.failed", "error"}:
                malformed = parse_malformed_tool_call_error(event)
                if malformed:
                    raise malformed
                raise RuntimeError(
                    event.get("message")
                    or event.get("error")
                    or str(event)
                )

    if final_response is None:
        raise IncompleteResponsesStreamError(
            "Responses stream ended without response.completed"
        )
    return final_response


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
    replay_guard: ResponsesStreamReplayGuard | None = None,
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

    reasoning_text = ""

    async def collect_text(snapshot: str) -> None:
        nonlocal text_buffer, last_draft, last_draft_text, draft_task
        text_buffer = snapshot
        if replay_guard is not None:
            replay_guard.observe(snapshot)
        now = time.monotonic()
        if (
            CHAT_STREAMING
            and now - last_draft >= CHAT_STREAM_INTERVAL
            and (draft_task is None or draft_task.done())
        ):
            last_draft_text = text_buffer
            draft_task = asyncio.create_task(publish_draft(last_draft_text))
            last_draft = now

    async def collect_reasoning(snapshot: str) -> None:
        nonlocal reasoning_text
        delta = (
            snapshot[len(reasoning_text):]
            if snapshot.startswith(reasoning_text)
            else snapshot
        )
        reasoning_text = snapshot
        if delta:
            if replay_guard is not None:
                replay_guard.observe(delta)
            await publish_reasoning_summary(delta)

    try:
        final_response = await stream_response_payload(
            url=f"{RESPONSES_BASE_URL}/responses",
            headers=auth_headers(),
            payload=payload,
            on_text=collect_text,
            on_reasoning_summary=collect_reasoning,
        )
    except BaseException:
        if draft_task is not None and not draft_task.done():
            draft_task.cancel()
        if draft_task is not None:
            await asyncio.gather(draft_task, return_exceptions=True)
        raise

    if (
        CHAT_STREAMING
        and text_buffer
    ):
        if draft_task is not None:
            await draft_task
        if text_buffer != last_draft_text:
            await publish_draft(text_buffer)

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
    replay_guard = ResponsesStreamReplayGuard()

    async def request() -> dict[str, Any]:
        try:
            return await _responses_create_stream_once(
                chat_id,
                input_items,
                tools=tools,
                extra_instructions=extra_instructions,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_summary=reasoning_summary,
                replay_guard=replay_guard,
            )
        except Exception as exc:
            replay_guard.reject_unsafe_replay(
                exc,
                operation="Responses stream",
            )
            raise

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
