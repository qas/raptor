"""Transient projection of root-agent tool activity."""

import asyncio
import json
import secrets
from collections import deque
from dataclasses import dataclass
from typing import Any

from chat_provider import (
    ActionButton,
    ConversationId,
    Controls,
    IncomingAction,
    MessageId,
    ProcessOutputChunk,
    ProcessOutputProvider,
    ToolConsoleProvider,
)
from chat_runtime import get_chat_provider
from chat_format import bash_console_block
from config import CHAT_STREAM_INTERVAL, MAX_TOOL_OUTPUT
from observability import log_exception


MAX_RETAINED_TOOL_BUBBLES = 64
MAX_CONSOLE_LINES = 7
MAX_CONSOLE_COMMAND_CHARS = 900
MAX_CONSOLE_LINE_CHARS = 300
_TOOL_VIEW_PREFIX = "toolview"


def tool_preview(call: dict[str, Any]) -> str:
    """Render the bounded call preview shared with tool approval."""
    name = str(call.get("name") or "unknown")
    raw_arguments = call.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
        rendered = json.dumps(
            arguments,
            indent=2,
            ensure_ascii=False,
        )
    except (TypeError, json.JSONDecodeError):
        rendered = str(raw_arguments)
    if len(rendered) > 3200:
        rendered = rendered[:3150] + "\n... [truncated]"
    return "Tool: " + name + "\n\n" + rendered


def _bounded_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3):]


def _shell_command(call: dict[str, Any]) -> str:
    try:
        arguments = json.loads(str(call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("command") or "")


@dataclass
class _ToolBubble:
    token: str
    call_id: str
    call: dict[str, Any]
    status: str
    controls: Controls = ()
    message_id: MessageId | None = None
    view: str = "info"
    output: str = ""
    pending: bool = False
    projection_task: asyncio.Task[None] | None = None


_tool_views: dict[str, tuple["ToolActivitySurface", _ToolBubble]] = {}


class ToolActivitySurface:
    """Project each tool call through one fresh canonical status bubble."""

    def __init__(
        self,
        conversation_id: ConversationId,
    ) -> None:
        self.conversation_id = conversation_id
        self._active: _ToolBubble | None = None
        self._completed: deque[_ToolBubble] = deque()
        self._by_call_id: dict[str, _ToolBubble] = {}
        self._closed = False

    @property
    def message_id(self) -> MessageId | None:
        return self._active.message_id if self._active is not None else None

    def _new_bubble(
        self,
        status: str,
        call: dict[str, Any],
    ) -> _ToolBubble:
        while True:
            token = secrets.token_hex(6)
            if token not in _tool_views:
                break
        bubble = _ToolBubble(
            token=token,
            call_id=str(call.get("call_id") or ""),
            call=dict(call),
            status=status,
        )
        self._active = bubble
        self._index_call(bubble)
        return bubble

    def _index_call(self, bubble: _ToolBubble) -> None:
        for call_id, candidate in tuple(self._by_call_id.items()):
            if candidate is bubble:
                self._by_call_id.pop(call_id, None)
        if bubble.call_id:
            self._by_call_id[bubble.call_id] = bubble

    def _update_active(
        self,
        status: str,
        call: dict[str, Any],
        controls: Controls = (),
    ) -> _ToolBubble:
        bubble = self._active or self._new_bubble(status, call)
        bubble.status = status
        bubble.call = dict(call)
        bubble.call_id = str(call.get("call_id") or "")
        bubble.controls = controls
        self._index_call(bubble)
        return bubble

    def _supports_console(self) -> bool:
        provider = get_chat_provider()
        return bool(
            isinstance(provider, ToolConsoleProvider)
            and provider.supports_tool_console(self.conversation_id)
        )

    @staticmethod
    def _console_controls(bubble: _ToolBubble) -> Controls:
        return ((
            ActionButton(
                "Info",
                f"{_TOOL_VIEW_PREFIX}:{bubble.token}:info",
            ),
            ActionButton(
                "Console",
                f"{_TOOL_VIEW_PREFIX}:{bubble.token}:console",
            ),
        ),)

    @staticmethod
    def _console_text(bubble: _ToolBubble) -> str:
        command = _bounded_tail(
            _shell_command(bubble.call),
            MAX_CONSOLE_COMMAND_CHARS,
        )
        lines = bubble.output.splitlines()[-MAX_CONSOLE_LINES:]
        output = "\n".join(
            _bounded_tail(line, MAX_CONSOLE_LINE_CHARS)
            for line in lines
        )
        return bash_console_block(command, output)

    def _render(self, bubble: _ToolBubble) -> tuple[str, Controls]:
        supports_console = (
            bubble.call.get("name") == "shell"
            and self._supports_console()
        )
        if supports_console:
            _tool_views[bubble.token] = (self, bubble)
        controls = bubble.controls
        if not controls and supports_console:
            controls = self._console_controls(bubble)
        if supports_console and bubble.view == "console":
            return self._console_text(bubble), controls
        return bubble.status + "\n\n" + tool_preview(bubble.call), controls

    async def _project(self, bubble: _ToolBubble) -> None:
        try:
            provider = get_chat_provider()
            text, controls = self._render(bubble)
            effective_controls = (
                controls if provider.capabilities.controls else ()
            )
            if bubble.message_id is not None:
                await provider.edit_message(
                    self.conversation_id,
                    bubble.message_id,
                    text,
                    effective_controls,
                )
            else:
                bubble.message_id = await provider.create_message(
                    self.conversation_id,
                    text,
                    effective_controls,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception("tool_activity", "projection_error", exc)

    async def _drain(self, bubble: _ToolBubble) -> None:
        first = True
        while bubble.pending:
            if not first:
                await asyncio.sleep(CHAT_STREAM_INTERVAL)
            first = False
            bubble.pending = False
            await self._project(bubble)

    def _queue(self, bubble: _ToolBubble) -> None:
        if self._closed:
            return
        bubble.pending = True
        task = bubble.projection_task
        if task is None or task.done():
            bubble.projection_task = asyncio.create_task(
                self._drain(bubble)
            )

    async def _flush(self, bubble: _ToolBubble) -> None:
        task = bubble.projection_task
        if task is not None:
            await task

    async def stream(
        self,
        call: dict[str, Any],
        complete: bool,
    ) -> None:
        bubble = self._update_active("Preparing tool", call)
        self._queue(bubble)
        if complete:
            await self._flush(bubble)

    async def running(self, call: dict[str, Any]) -> None:
        bubble = self._update_active("Running", call)
        self._queue(bubble)
        await self._flush(bubble)

    async def approval(
        self,
        call: dict[str, Any],
        controls: Controls,
    ) -> int | str:
        bubble = self._update_active(
            "⚠️ Approval required",
            call,
            controls,
        )
        self._queue(bubble)
        await self._flush(bubble)
        if bubble.message_id is None:
            raise RuntimeError("Could not present tool approval")
        return bubble.message_id

    async def publish_process_output(
        self,
        chunk: ProcessOutputChunk,
    ) -> None:
        bubble = self._by_call_id.get(chunk.call_id)
        if bubble is not None:
            addition = chunk.text
            if chunk.truncated:
                addition += "\n... [output truncated] ..."
            bubble.output = (bubble.output + addition)[-MAX_TOOL_OUTPUT:]
            if bubble.view == "console":
                self._queue(bubble)
        provider = get_chat_provider()
        if isinstance(provider, ProcessOutputProvider):
            await provider.publish_process_output(self.conversation_id, chunk)

    async def select_view(
        self,
        bubble: _ToolBubble,
        *,
        conversation_id: ConversationId,
        message_id: MessageId | None,
        view: str,
    ) -> bool:
        if (
            self._closed
            or conversation_id != self.conversation_id
            or message_id != bubble.message_id
            or view not in {"info", "console"}
        ):
            return False
        bubble.view = view
        self._queue(bubble)
        await self._flush(bubble)
        return True

    async def _delete_messages(
        self,
        message_ids: tuple[MessageId, ...],
    ) -> None:
        if not message_ids:
            return
        try:
            await get_chat_provider().delete_messages(
                self.conversation_id,
                message_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception("tool_activity", "bubble_cleanup_error", exc)

    def _discard(self, bubble: _ToolBubble) -> None:
        _tool_views.pop(bubble.token, None)
        for call_id, candidate in tuple(self._by_call_id.items()):
            if candidate is bubble:
                self._by_call_id.pop(call_id, None)

    async def finished(
        self,
        call: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        status = "Completed" if result.get("ok") else "Failed"
        if result.get("status") == "interrupted":
            status = "Interrupted"
        if result.get("approval") == "denied":
            status = "Denied"
        elif result.get("approval") == "superseded":
            status = "Interrupted"
        bubble = self._update_active(status, call)
        self._queue(bubble)
        await self._flush(bubble)
        if bubble.message_id is not None:
            if len(self._completed) >= MAX_RETAINED_TOOL_BUBBLES:
                oldest = self._completed.popleft()
                await self._delete_messages((oldest.message_id,))
                self._discard(oldest)
            self._completed.append(bubble)
        else:
            self._discard(bubble)
        self._active = None

    async def clear(self) -> None:
        self._closed = True
        bubbles = list(self._completed)
        if self._active is not None:
            bubbles.append(self._active)
        tasks = [
            bubble.projection_task
            for bubble in bubbles
            if bubble.projection_task is not None
            and not bubble.projection_task.done()
        ]
        for bubble in bubbles:
            bubble.pending = False
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        message_ids = tuple(
            bubble.message_id
            for bubble in reversed(bubbles)
            if bubble.message_id is not None
        )
        await self._delete_messages(message_ids)
        for bubble in bubbles:
            self._discard(bubble)
        self._active = None
        self._completed.clear()


async def handle_tool_activity_action(action: IncomingAction) -> bool:
    parts = action.data.split(":")
    if not parts or parts[0] != _TOOL_VIEW_PREFIX:
        return False
    provider = get_chat_provider()
    selected = False
    if len(parts) == 3:
        token, view = parts[1:]
        entry = _tool_views.get(token)
        if entry is not None:
            surface, bubble = entry
            conversation_id = (
                action.presentation_conversation_id
                or action.conversation_id
            )
            if conversation_id is not None:
                selected = await surface.select_view(
                    bubble,
                    conversation_id=conversation_id,
                    message_id=action.message_id,
                    view=view,
                )
    try:
        await provider.answer_action(
            action.action_id,
            "" if selected else "Tool view is no longer available.",
        )
    except Exception as exc:
        log_exception(provider.name, "action_answer_error", exc)
    return True
