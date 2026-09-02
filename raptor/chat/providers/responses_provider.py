"""Lightweight OpenAI Responses-compatible inbound chat provider."""
import asyncio
import contextvars
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from raptor.chat.chat_provider import (
    ChatEvent,
    ConversationId,
    Controls,
    IncomingAction,
    IncomingMessage,
    PollResult,
    ProviderCapabilities,
)
from raptor.config import (
    RESPONSES_SERVER_API_KEY,
    RESPONSES_SERVER_HOST,
    RESPONSES_SERVER_MAX_BODY,
    RESPONSES_SERVER_MAX_CONNECTIONS,
    RESPONSES_SERVER_MAX_PENDING,
    RESPONSES_SERVER_MAX_STATUS_MESSAGES,
    RESPONSES_SERVER_MAX_STREAM_EVENTS,
    RESPONSES_SERVER_PORT,
    RESPONSES_SERVER_READ_TIMEOUT,
)
from raptor.observability import log_exception

SSE_HEARTBEAT_SECONDS = 10.0
CONVERSATION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "responses_provider_request_id",
    default=None,
)


@dataclass
class PendingResponse:
    response_id: str
    request_id: str
    conversation_id: str
    stream: bool
    created_at: int = field(default_factory=lambda: int(time.time()))
    completed: asyncio.Future[dict[str, Any]] | None = None
    events: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=256),
    )
    streamed_text: str = ""
    reasoning_summary: str = ""
    reasoning_item_id: str = field(
        default_factory=lambda: "rs_" + secrets.token_hex(12),
    )
    action: bool = False
    action_result: dict[str, Any] | None = None
    delivery_captured: bool = False


@dataclass(frozen=True)
class QueuedEvent:
    request_id: str
    event: ChatEvent


class ProviderOverloadedError(RuntimeError):
    """The inbound provider reached its explicit request capacity."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"}:
            text = part.get("text")
            if text is not None:
                parts.append(str(text))
    return "\n".join(parts)


def input_text(value: Any) -> str:
    """Return only the newest user turn from Responses-style input."""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        raise ValueError("input must be a string or a list of messages")
    latest = ""
    for item in value:
        if isinstance(item, str):
            if item.strip():
                latest = item
            continue
        if not isinstance(item, dict):
            continue
        if item.get("role") not in {None, "user"}:
            continue
        text = _content_text(item.get("content"))
        if not text and item.get("type") in {"input_text", "text"}:
            text = str(item.get("text") or "")
        if text:
            latest = text
    text = latest.strip()
    if not text:
        raise ValueError("input contains no user text")
    return text


class ResponsesApiProvider:
    """Expose named durable conversations over HTTP and SSE."""

    name = "responses_api"
    authorized_user_id = "api:operator"
    primary_conversation_id = "default"
    capabilities = ProviderCapabilities(
        drafts=True,
        reasoning_summaries=True,
        pins=True,
        controls=True,
        typing=True,
    )

    def __init__(
        self,
        *,
        host: str = RESPONSES_SERVER_HOST,
        port: int = RESPONSES_SERVER_PORT,
        api_key: str = RESPONSES_SERVER_API_KEY,
        max_body: int = RESPONSES_SERVER_MAX_BODY,
        max_connections: int = RESPONSES_SERVER_MAX_CONNECTIONS,
        max_pending: int = RESPONSES_SERVER_MAX_PENDING,
        max_status_messages: int = RESPONSES_SERVER_MAX_STATUS_MESSAGES,
        max_stream_events: int = RESPONSES_SERVER_MAX_STREAM_EVENTS,
        read_timeout: float = RESPONSES_SERVER_READ_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self.max_body = max_body
        self.max_connections = max_connections
        self.max_pending = max_pending
        self.max_status_messages = max_status_messages
        self.max_stream_events = max_stream_events
        self.read_timeout = read_timeout
        self.server: asyncio.AbstractServer | None = None
        self.events: asyncio.Queue[QueuedEvent] = asyncio.Queue(
            maxsize=max_pending,
        )
        self.pending: dict[str, PendingResponse] = {}
        self.messages: dict[str, dict[str, Any]] = {}
        self.inbox: dict[str, dict[str, dict[str, Any]]] = {}
        self.event_requests: dict[int, str] = {}
        self._cursor = 0
        self._message_counter = 0
        self._active_connections = 0

    @property
    def bound_port(self) -> int:
        if self.server and self.server.sockets:
            return int(self.server.sockets[0].getsockname()[1])
        return self.port

    async def initialize(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        del commands
        if self.host not in {"127.0.0.1", "::1", "localhost"} and not self.api_key:
            raise RuntimeError(
                "RESPONSES_SERVER_API_KEY is required when "
                "RESPONSES_SERVER_HOST is not loopback"
            )
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        for pending in tuple(self.pending.values()):
            if pending.completed and not pending.completed.done():
                pending.completed.set_exception(
                    RuntimeError("Responses API provider closed"),
                )
        self.pending.clear()
        self.inbox.clear()
        self.event_requests.clear()
        while True:
            try:
                self.events.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.events.task_done()
        self._clear_request_context()

    async def poll(
        self,
        cursor: object | None,
        *,
        timeout: int,
    ) -> PollResult:
        del timeout
        queued = await self.events.get()
        self.events.task_done()
        self._cursor += 1
        return PollResult((queued.event,), self._cursor)

    @staticmethod
    def encode_conversation_id(conversation_id: ConversationId) -> str:
        return str(conversation_id)

    @staticmethod
    def decode_conversation_id(value: str) -> str:
        return value

    def prepare_event(self, event: ChatEvent) -> None:
        """Restore request correlation immediately before event handling."""
        request_id = self.event_requests.pop(id(event), None)
        _request_id.set(request_id)

    def _clear_request_context(self) -> None:
        _request_id.set(None)

    def _pending(self) -> PendingResponse | None:
        request_id = _request_id.get()
        return self.pending.get(request_id or "")

    def _conversation_pending(
        self,
        conversation_id: ConversationId,
    ) -> PendingResponse | None:
        pending = self._pending()
        if (
            pending is not None
            and pending.conversation_id != str(conversation_id)
        ):
            raise ValueError("delivery context belongs to another conversation")
        return pending

    def capture_delivery_context(
        self,
        conversation_id: ConversationId,
    ) -> str | None:
        pending = self._conversation_pending(conversation_id)
        if pending is not None:
            pending.delivery_captured = True
        return _request_id.get()

    def activate_delivery_context(
        self,
        conversation_id: ConversationId,
        value: Any | None,
    ) -> contextvars.Token[str | None]:
        request_id = str(value) if value is not None else None
        pending = self.pending.get(request_id or "")
        if (
            pending is not None
            and pending.conversation_id != str(conversation_id)
        ):
            raise ValueError("delivery context belongs to another conversation")
        return _request_id.set(request_id)

    def restore_delivery_context(
        self,
        token: contextvars.Token[str | None],
    ) -> None:
        _request_id.reset(token)

    def _emit(self, event: dict[str, Any]) -> None:
        pending = self._pending()
        if pending is not None and pending.stream:
            self._emit_stream_event(pending, event)

    @staticmethod
    def _emit_stream_event(
        pending: PendingResponse,
        event: dict[str, Any],
        *,
        terminal: bool = False,
    ) -> None:
        """Bound streamed projections while preserving terminal state."""
        if pending.events.full():
            if not terminal:
                return
            try:
                pending.events.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                pending.events.task_done()
        pending.events.put_nowait(event)

    @staticmethod
    def _actions(controls: Controls) -> list[dict[str, str]]:
        return [
            {"label": button.label, "data": button.action}
            for row in controls
            for button in row
        ]

    def _status_event(
        self,
        operation: str,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "raptor.status",
            "operation": operation,
            "conversation_id": message["conversation_id"],
            "message_id": message["message_id"],
            "text": message["text"],
            "actions": message["actions"],
            "pinned": message["pinned"],
        }

    def _store_message(self, message_id: str, message: dict[str, Any]) -> None:
        self.messages[message_id] = message
        conversation_id = message["conversation_id"]
        while True:
            matching = [
                key
                for key, item in self.messages.items()
                if item["conversation_id"] == conversation_id
            ]
            if len(matching) <= self.max_status_messages:
                return
            evictable = next(
                (
                    key
                    for key in matching
                    if not self.messages[key]["pinned"]
                    and not self.messages[key]["actions"]
                ),
                matching[0],
            )
            self.messages.pop(evictable, None)

    def _store_inbox_message(
        self,
        conversation_id: ConversationId,
        text: str,
    ) -> None:
        conversation = str(conversation_id)
        self._message_counter += 1
        message_id = f"inbox_{self._message_counter}"
        messages = self.inbox.setdefault(conversation, {})
        messages[message_id] = {
            "conversation_id": conversation,
            "message_id": message_id,
            "text": str(text),
            "actions": [],
            "pinned": False,
            "asynchronous": True,
        }
        while len(messages) > self.max_status_messages:
            messages.pop(next(iter(messages)), None)

    def _response(self, pending: PendingResponse, text: str) -> dict[str, Any]:
        message_id = "msg_" + secrets.token_hex(12)
        output: list[dict[str, Any]] = [{
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        }]
        if pending.reasoning_summary:
            output.append({
                "id": pending.reasoning_item_id,
                "type": "reasoning",
                "summary": [{
                    "type": "summary_text",
                    "text": pending.reasoning_summary,
                }],
            })
        return {
            "id": pending.response_id,
            "object": "response",
            "created_at": pending.created_at,
            "status": "completed",
            "model": "raptor",
            "output": output,
            "output_text": text,
            "error": None,
            "incomplete_details": None,
        }

    async def send_text(self, conversation_id, text: str) -> tuple[str, ...]:
        pending = self._conversation_pending(conversation_id)
        if pending is None:
            self._store_inbox_message(conversation_id, str(text))
            return ()
        response = self._response(pending, str(text))
        if pending.completed is not None and not pending.completed.done():
            pending.completed.set_result(response)
        final_text = str(text)
        if pending.stream and final_text.startswith(pending.streamed_text):
            delta = final_text[len(pending.streamed_text):]
            if delta:
                self._emit_stream_event(pending, {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                })
        if pending.stream:
            if pending.reasoning_summary:
                self._emit_stream_event(pending, {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": pending.reasoning_item_id,
                    "output_index": 1,
                    "summary_index": 0,
                    "text": pending.reasoning_summary,
                }, terminal=True)
            self._emit_stream_event(pending, {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": final_text,
            }, terminal=True)
            self._emit_stream_event(pending, {
                "type": "response.completed",
                "response": response,
            }, terminal=True)
        return ()

    async def send_draft(
        self,
        conversation_id,
        draft_id: int,
        text: str,
    ) -> None:
        del draft_id
        pending = self._conversation_pending(conversation_id)
        if pending is None or not pending.stream:
            return
        current = str(text)
        if current.startswith(pending.streamed_text):
            delta = current[len(pending.streamed_text):]
        else:
            delta = current
        pending.streamed_text = current
        if delta:
            self._emit_stream_event(pending, {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
            })

    async def send_reasoning_summary(
        self,
        conversation_id,
        delta: str,
    ) -> None:
        pending = self._conversation_pending(conversation_id)
        text = str(delta)
        if pending is None or not pending.stream or not text:
            return
        pending.reasoning_summary += text
        self._emit_stream_event(pending, {
            "type": "response.reasoning_summary_text.delta",
            "item_id": pending.reasoning_item_id,
            "output_index": 1,
            "summary_index": 0,
            "delta": text,
        })

    async def create_message(
        self,
        conversation_id,
        text: str,
        controls: Controls = (),
    ) -> str:
        self._conversation_pending(conversation_id)
        self._message_counter += 1
        message_id = f"status_{self._message_counter}"
        message = {
            "conversation_id": str(conversation_id),
            "message_id": message_id,
            "text": str(text),
            "actions": self._actions(controls),
            "pinned": False,
        }
        self._store_message(message_id, message)
        self._emit(self._status_event("created", message))
        return message_id

    async def edit_message(
        self,
        conversation_id,
        message_id,
        text: str,
        controls: Controls = (),
    ) -> None:
        self._conversation_pending(conversation_id)
        key = str(message_id)
        previous = self.messages.get(key, {})
        if (
            previous
            and previous["conversation_id"] != str(conversation_id)
        ):
            raise ValueError("message belongs to another conversation")
        message = {
            "conversation_id": str(conversation_id),
            "message_id": key,
            "text": str(text),
            "actions": self._actions(controls),
            "pinned": bool(previous.get("pinned", False)),
        }
        self._store_message(key, message)
        self._emit(self._status_event("updated", message))

    async def delete_message(self, conversation_id, message_id) -> None:
        self._conversation_pending(conversation_id)
        key = str(message_id)
        message = self.messages.get(key)
        if (
            message is not None
            and message["conversation_id"] != str(conversation_id)
        ):
            raise ValueError("message belongs to another conversation")
        self.messages.pop(key, None)
        self._emit({
            "type": "raptor.status",
            "operation": "deleted",
            "conversation_id": str(conversation_id),
            "message_id": key,
        })

    async def delete_messages(self, conversation_id, message_ids) -> None:
        self._conversation_pending(conversation_id)
        keys = tuple(str(message_id) for message_id in message_ids)
        for key in keys:
            message = self.messages.get(key)
            if (
                message is not None
                and message["conversation_id"] != str(conversation_id)
            ):
                raise ValueError("message belongs to another conversation")
        for key in keys:
            self.messages.pop(key, None)
            self._emit({
                "type": "raptor.status",
                "operation": "deleted",
                "conversation_id": str(conversation_id),
                "message_id": key,
            })

    async def pin_message(self, conversation_id, message_id) -> None:
        self._conversation_pending(conversation_id)
        message = self.messages.get(str(message_id))
        if message is None:
            return
        if message["conversation_id"] != str(conversation_id):
            raise ValueError("message belongs to another conversation")
        message["pinned"] = True
        self._emit(self._status_event("pinned", message))

    async def unpin_message(self, conversation_id, message_id) -> None:
        self._conversation_pending(conversation_id)
        message = self.messages.get(str(message_id))
        if message is None:
            return
        if message["conversation_id"] != str(conversation_id):
            raise ValueError("message belongs to another conversation")
        message["pinned"] = False
        self._emit(self._status_event("unpinned", message))

    async def set_typing(self, conversation_id, active: bool) -> None:
        self._conversation_pending(conversation_id)
        self._emit({
            "type": "raptor.status",
            "operation": "typing",
            "conversation_id": str(conversation_id),
            "active": bool(active),
        })

    async def reject_busy_message(self, conversation_id) -> bool:
        self._conversation_pending(conversation_id)
        return False

    async def acknowledge_queued_message(self, conversation_id) -> None:
        self._conversation_pending(conversation_id)
        # Request-style steering remains open until the queued turn produces
        # its real response. The core restores this request's delivery context
        # when the work is selected.

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None:
        pending = self._pending()
        if pending is None or not pending.action:
            return
        pending.action_result = {
            "id": action_id,
            "object": "raptor.action",
            "status": "accepted",
            "message": text,
            "alert": alert,
        }

    async def finish_event(self, event: ChatEvent) -> None:
        del event
        pending = self._pending()
        try:
            if (
                pending is not None
                and pending.action
                and pending.completed is not None
                and not pending.completed.done()
            ):
                pending.completed.set_result(
                    pending.action_result
                    or {
                        "id": pending.response_id,
                        "object": "raptor.action",
                        "status": "accepted",
                        "message": "",
                        "alert": False,
                    }
                )
            elif (
                pending is not None
                and pending.completed is not None
                and not pending.completed.done()
                and not pending.delivery_captured
            ):
                pending.completed.set_exception(
                    RuntimeError("request ended without a response")
                )
        finally:
            self._clear_request_context()

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, dict[str, str], dict[str, str], bytes]:
        header = await reader.readuntil(b"\r\n\r\n")
        if len(header) > 65536:
            raise ValueError("request headers are too large")
        lines = header[:-4].decode("latin-1").split("\r\n")
        method, target, _version = lines[0].split(" ", 2)
        url = urlsplit(target)
        query = {
            key: values[-1]
            for key, values in parse_qs(url.query, keep_blank_values=True).items()
        }
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if separator:
                headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length < 0 or length > self.max_body:
            raise ValueError("request body is too large")
        body = await reader.readexactly(length) if length else b""
        return method.upper(), url.path, query, headers, body

    def _authorized(self, headers: dict[str, str]) -> bool:
        if not self.api_key:
            return True
        provided = headers.get("authorization", "")
        expected = f"Bearer {self.api_key}"
        return hmac.compare_digest(provided, expected)

    async def _write_json(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: Any,
    ) -> None:
        encoded = _json_bytes(body)
        reasons = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            408: "Request Timeout",
            429: "Too Many Requests",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }
        header = (
            f"HTTP/1.1 {status} {reasons.get(status, 'Error')}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(encoded)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(header + encoded)
        await writer.drain()

    async def _write_sse(
        self,
        writer: asyncio.StreamWriter,
        pending: PendingResponse,
    ) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        created = {
            "type": "response.created",
            "response": {
                "id": pending.response_id,
                "object": "response",
                "created_at": pending.created_at,
                "status": "in_progress",
                "output": [],
            },
        }
        await self._sse_event(writer, created)
        while True:
            try:
                event = await asyncio.wait_for(
                    pending.events.get(),
                    timeout=SSE_HEARTBEAT_SECONDS,
                )
            except asyncio.TimeoutError:
                writer.write(b": keep-alive\n\n")
                await writer.drain()
                continue
            pending.events.task_done()
            await self._sse_event(writer, event)
            if event.get("type") in {"response.completed", "response.failed"}:
                break

    async def _sse_event(
        self,
        writer: asyncio.StreamWriter,
        event: dict[str, Any],
    ) -> None:
        writer.write(
            f"event: {event.get('type', 'message')}\n".encode("utf-8")
            + b"data: "
            + _json_bytes(event)
            + b"\n\n"
        )
        await writer.drain()

    async def _queue_message(
        self,
        payload: dict[str, Any],
    ) -> PendingResponse:
        if len(self.pending) >= self.max_pending:
            raise ProviderOverloadedError("too many pending requests")
        conversation = payload.get("conversation")
        if conversation is not None and not isinstance(conversation, str):
            raise ValueError("conversation must be a string")
        conversation_id = conversation or self.primary_conversation_id
        if not CONVERSATION_PATTERN.fullmatch(conversation_id):
            raise ValueError(
                "conversation must be 1-128 letters, numbers, dots, "
                "underscores, or hyphens"
            )
        text = input_text(payload.get("input"))
        request_id = "req_" + secrets.token_hex(12)
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("model must be a string")
        pending = PendingResponse(
            response_id="resp_" + secrets.token_hex(12),
            request_id=request_id,
            conversation_id=conversation_id,
            stream=stream,
            completed=(
                None
                if stream
                else asyncio.get_running_loop().create_future()
            ),
            events=asyncio.Queue(maxsize=self.max_stream_events),
        )
        self.pending[request_id] = pending
        event = IncomingMessage(
            conversation_id=conversation_id,
            sender_id=self.authorized_user_id,
            message_id="in_" + secrets.token_hex(12),
            text=text,
        )
        self.event_requests[id(event)] = request_id
        try:
            self.events.put_nowait(QueuedEvent(request_id, event))
        except asyncio.QueueFull as exc:
            self.pending.pop(request_id, None)
            self.event_requests.pop(id(event), None)
            raise ProviderOverloadedError(
                "too many pending requests"
            ) from exc
        return pending

    async def _queue_action(self, payload: dict[str, Any]) -> PendingResponse:
        if len(self.pending) >= self.max_pending:
            raise ProviderOverloadedError("too many pending requests")
        data_value = payload.get("data")
        if not isinstance(data_value, str):
            raise ValueError("data must be a string")
        data = data_value.strip()
        if not data:
            raise ValueError("data is required")
        conversation = payload.get("conversation")
        if conversation is not None and not isinstance(conversation, str):
            raise ValueError("conversation must be a string")
        conversation_id = conversation or self.primary_conversation_id
        if not CONVERSATION_PATTERN.fullmatch(conversation_id):
            raise ValueError("invalid conversation")
        request_id = "req_" + secrets.token_hex(12)
        pending = PendingResponse(
            response_id="action_" + secrets.token_hex(12),
            request_id=request_id,
            conversation_id=conversation_id,
            stream=False,
            completed=asyncio.get_running_loop().create_future(),
            events=asyncio.Queue(maxsize=self.max_stream_events),
            action=True,
        )
        self.pending[request_id] = pending
        event = IncomingAction(
            action_id=pending.response_id,
            conversation_id=conversation_id,
            sender_id=self.authorized_user_id,
            message_id=payload.get("message_id"),
            data=data,
        )
        self.event_requests[id(event)] = request_id
        try:
            self.events.put_nowait(QueuedEvent(request_id, event))
        except asyncio.QueueFull as exc:
            self.pending.pop(request_id, None)
            self.event_requests.pop(id(event), None)
            raise ProviderOverloadedError(
                "too many pending requests"
            ) from exc
        return pending

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._active_connections >= self.max_connections:
            try:
                await self._write_json(writer, 503, {
                    "error": {
                        "code": "server_overloaded",
                        "message": "Server is at connection capacity",
                        "type": "server_error",
                    }
                })
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, BrokenPipeError):
                    pass
            return
        self._active_connections += 1
        try:
            await self._serve_connection(reader, writer)
        finally:
            self._active_connections -= 1

    async def _serve_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        pending: PendingResponse | None = None
        correlation_id = "req_" + secrets.token_hex(12)
        try:
            method, path, query, headers, body = await asyncio.wait_for(
                self._read_request(reader),
                timeout=self.read_timeout,
            )
            if not self._authorized(headers):
                await self._write_json(
                    writer,
                    401,
                    {
                        "error": {
                            "message": "Unauthorized",
                            "type": "authentication_error",
                        }
                    },
                )
                return
            if method == "GET" and path == "/healthz":
                await self._write_json(writer, 200, {"ok": True})
                return
            if method == "GET" and path == "/v1/status":
                conversation_id = query.get(
                    "conversation",
                    self.primary_conversation_id,
                )
                if not CONVERSATION_PATTERN.fullmatch(conversation_id):
                    raise ValueError("invalid conversation")
                await self._write_json(writer, 200, {
                    "object": "raptor.status.list",
                    "conversation": conversation_id,
                    "data": [
                        message
                        for message in self.messages.values()
                        if message["conversation_id"] == conversation_id
                    ] + list(self.inbox.get(conversation_id, {}).values()),
                })
                return
            if method == "GET" and path == "/v1/models":
                await self._write_json(writer, 200, {
                    "object": "list",
                    "data": [{
                        "id": "raptor",
                        "object": "model",
                        "created": 0,
                        "owned_by": "raptor",
                    }],
                })
                return
            if method != "POST":
                await self._write_json(
                    writer,
                    405,
                    {
                        "error": {
                            "message": "Method not allowed",
                            "type": "invalid_request_error",
                        }
                    },
                )
                return
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            if path == "/v1/responses":
                pending = await self._queue_message(payload)
            elif path == "/v1/actions":
                pending = await self._queue_action(payload)
            else:
                await self._write_json(writer, 404, {
                    "error": {"message": "Not found", "type": "invalid_request_error"},
                })
                return
            if pending.stream:
                await self._write_sse(writer, pending)
            else:
                assert pending.completed is not None
                result = await pending.completed
                await self._write_json(writer, 200, result)
        except asyncio.TimeoutError:
            await self._write_json(writer, 408, {
                "error": {
                    "code": "request_timeout",
                    "message": "Request headers or body timed out",
                    "type": "invalid_request_error",
                }
            })
        except ProviderOverloadedError:
            await self._write_json(writer, 429, {
                "error": {
                    "code": "agent_overloaded",
                    "message": "Too many pending requests",
                    "type": "rate_limit_error",
                }
            })
        except (
            ValueError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ) as exc:
            await self._write_json(writer, 400, {
                "error": {"message": str(exc), "type": "invalid_request_error"},
            })
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as exc:
            log_exception(
                "responses_api",
                "request_error",
                exc,
                {"request_id": correlation_id},
            )
            try:
                await self._write_json(writer, 500, {
                    "error": {
                        "code": "internal_error",
                        "message": "Internal server error",
                        "request_id": correlation_id,
                        "type": "server_error",
                    },
                })
            except Exception:
                pass
        finally:
            if pending is not None:
                self.pending.pop(pending.request_id, None)
                self.event_requests = {
                    event_id: request_id
                    for event_id, request_id in self.event_requests.items()
                    if request_id != pending.request_id
                }
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


responses_provider = ResponsesApiProvider()
