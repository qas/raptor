"""Transient projection of root-agent tool activity."""

import asyncio
import json
import math
import secrets
from collections import deque
from dataclasses import dataclass
from typing import Any

from raptor.chat.chat_format import bash_console_block
from raptor.chat.chat_provider import (
    ActionButton,
    ConversationId,
    Controls,
    IncomingAction,
    MessageId,
    ProcessOutputChunk,
    ProcessOutputProvider,
    ToolConsoleProvider,
)
from raptor.chat.chat_runtime import get_chat_provider
from raptor.config import CHAT_STREAM_INTERVAL, CHAT_TOOL_ACTIVITY, MAX_TOOL_OUTPUT
from raptor.observability import log_exception


MAX_RETAINED_TOOL_BUBBLES = 64
MAX_CONSOLE_LINES = 7
MAX_CONSOLE_COMMAND_CHARS = 900
MAX_CONSOLE_LINE_CHARS = 300
MAX_TOOL_NAME_CHARS = 128
MAX_TOOL_PREVIEW_CHARS = 3200
WAIT_UPDATE_INTERVAL_SECONDS = 5
_TOOL_VIEW_PREFIX = "toolview"


def _field_label(key: object) -> str:
    normalized = "".join(
        character if character.isalnum() else " "
        for character in str(key)
    )
    words = normalized.split()
    if not words:
        return "Value"
    acronyms = {"api", "id", "uri", "url"}
    rendered = [
        word.upper() if word.lower() in acronyms else word.lower()
        for word in words
    ]
    if words[0].lower() not in acronyms:
        rendered[0] = rendered[0].capitalize()
    if len(rendered) > 1 and words[-1].lower() == "ms":
        rendered[-1] = "(ms)"
    return " ".join(rendered)


def _fenced_value(value: str, language: str, limit: int) -> str:
    opening = "```" + language + "\n"
    closing = "\n```"
    safe = value.replace("```", "``\u200b`")
    if len(opening) + len(safe) + len(closing) <= limit:
        return opening + safe + closing
    suffix = "\n... [truncated]"
    keep = max(0, limit - len(opening) - len(suffix) - len(closing))
    return opening + safe[:keep] + suffix + closing


def _render_argument_value(value: object, limit: int) -> str:
    if isinstance(value, str):
        if "\n" not in value and "`" not in value:
            inline = "`" + value + "`"
            if len(inline) <= limit:
                return inline
        return _fenced_value(value, "text", limit)
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return "`" + str(value).lower() + "`"
    if isinstance(value, (int, float)):
        return "`" + str(value) + "`"
    rendered = json.dumps(value, indent=2, ensure_ascii=False)
    return _fenced_value(rendered, "json", limit)


def _render_arguments(arguments: object, limit: int) -> str:
    if not isinstance(arguments, dict):
        rendered = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, indent=2, ensure_ascii=False)
        )
        return "**Arguments:**\n" + _fenced_value(
            str(rendered),
            "json",
            max(32, limit - len("**Arguments:**\n")),
        )
    if not arguments:
        return "**Arguments:** `none`"
    sections: list[str] = []
    used = 0
    for key, value in arguments.items():
        label = "**" + _field_label(key) + ":**"
        separator = "\n\n" if sections else ""
        minimum = len(separator) + len(label) + 1 + 32
        if limit - used < minimum:
            omission = separator + "... [additional fields omitted]"
            if used + len(omission) <= limit:
                sections.append(omission)
            break
        available = limit - used - len(separator) - len(label) - 1
        rendered = _render_argument_value(value, available)
        section = separator + label
        section += " " if not rendered.startswith("```") else "\n"
        section += rendered
        sections.append(section)
        used += len(section)
    return "".join(sections)


def tool_preview(call: dict[str, Any]) -> str:
    """Render the bounded call preview shared with tool approval."""
    name = str(call.get("name") or "unknown")
    if len(name) > MAX_TOOL_NAME_CHARS:
        name = name[:MAX_TOOL_NAME_CHARS - 3] + "..."
    raw_arguments = call.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        arguments = str(raw_arguments)
    heading = "**Tool:** `" + name.replace("`", "") + "`"
    remaining = MAX_TOOL_PREVIEW_CHARS - len(heading) - 2
    return heading + "\n\n" + _render_arguments(arguments, remaining)


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


def _poll_wait_ms(call: dict[str, Any]) -> int | None:
    if call.get("name") != "write_stdin":
        return None
    try:
        arguments = json.loads(str(call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict) or str(arguments.get("chars") or ""):
        return None
    try:
        from raptor.shell.shell_sessions import write_stdin_wait_ms

        return write_stdin_wait_ms(arguments)
    except (TypeError, ValueError):
        return None


def _wait_duration(seconds: int) -> str:
    minutes, remaining = divmod(max(0, seconds), 60)
    if minutes and remaining:
        return f"{minutes}m {remaining}s"
    if minutes:
        return f"{minutes}m"
    return f"{remaining}s"


@dataclass
class _ToolBubble:
    token: str
    call_id: str
    call: dict[str, Any]
    status: str
    controls: Controls = ()
    message_id: MessageId | None = None
    view: str = "console"
    output: str = ""
    pending: bool = False
    projection_task: asyncio.Task[None] | None = None
    wait_seconds: int | None = None
    wait_deadline: float | None = None
    wait_task: asyncio.Task[None] | None = None
    wait_terminal: str | None = None
    console_available: bool = True
    terminal_info: str | None = None


_tool_views: dict[str, tuple["ToolActivitySurface", _ToolBubble]] = {}


class ToolActivitySurface:
    """Project each tool call through one fresh canonical status bubble."""

    def __init__(
        self,
        conversation_id: ConversationId,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._enabled = CHAT_TOOL_ACTIVITY if enabled is None else enabled
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
        view = "info" if bubble.view == "console" else "console"
        button = ActionButton(
            view.title(),
            f"{_TOOL_VIEW_PREFIX}:{bubble.token}:{view}",
        )
        return ((button,),)

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
            and bubble.console_available
            and self._supports_console()
        )
        if supports_console:
            _tool_views[bubble.token] = (self, bubble)
        controls = bubble.controls
        if not controls and supports_console:
            controls = self._console_controls(bubble)
        if (
            supports_console
            and not bubble.controls
            and bubble.view == "console"
        ):
            return self._console_text(bubble), controls
        if bubble.terminal_info is not None:
            return bubble.terminal_info, controls
        if bubble.wait_terminal is not None:
            return bubble.wait_terminal, controls
        if bubble.wait_seconds is not None:
            return "Waiting " + _wait_duration(bubble.wait_seconds), controls
        return bubble.status + "\n\n" + tool_preview(bubble.call), controls

    async def _animate_wait(self, bubble: _ToolBubble) -> None:
        deadline = bubble.wait_deadline
        if deadline is None:
            return
        loop = asyncio.get_running_loop()
        while not self._closed:
            delay = min(
                WAIT_UPDATE_INTERVAL_SECONDS,
                max(0.0, deadline - loop.time()),
            )
            if delay:
                await asyncio.sleep(delay)
            remaining = max(0, math.ceil(deadline - loop.time()))
            if remaining == bubble.wait_seconds:
                if not remaining:
                    return
                continue
            bubble.wait_seconds = remaining
            self._queue(bubble)
            if not remaining:
                await self._flush(bubble)
                return

    async def _stop_wait(self, bubble: _ToolBubble) -> None:
        task = bubble.wait_task
        bubble.wait_task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

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
        if not self._enabled:
            return
        bubble = self._update_active("Preparing tool", call)
        self._queue(bubble)
        if complete:
            await self._flush(bubble)

    async def running(self, call: dict[str, Any]) -> None:
        if not self._enabled and self._active is None:
            return
        bubble = self._update_active("Running", call)
        wait_ms = _poll_wait_ms(call)
        if wait_ms is not None and self._supports_console():
            bubble.wait_seconds = math.ceil(wait_ms / 1000)
        self._queue(bubble)
        await self._flush(bubble)
        if bubble.wait_seconds is not None and wait_ms is not None:
            loop = asyncio.get_running_loop()
            bubble.wait_deadline = loop.time() + wait_ms / 1000
            bubble.wait_task = asyncio.create_task(
                self._animate_wait(bubble)
            )

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
            or (view == "console" and not bubble.console_available)
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
        if not self._enabled and self._active is None:
            return
        status = "Completed" if result.get("ok") else "Failed"
        if result.get("status") == "interrupted":
            status = "Interrupted"
        if result.get("approval") == "denied":
            status = "Failed"
        elif result.get("approval") == "superseded":
            status = "Interrupted"
        bubble = self._update_active(status, call)
        if result.get("approval") in {"denied", "superseded"}:
            bubble.view = "info"
            bubble.console_available = False
            outcome = (
                "Command denied"
                if result.get("approval") == "denied"
                else "Command superseded"
            )
            bubble.terminal_info = (
                status + "\n\nTool: "
                + str(call.get("name") or "unknown")
                + "\n\n"
                + outcome
            )
        await self._stop_wait(bubble)
        if bubble.wait_seconds is not None:
            result_status = str(result.get("status") or "")
            if result_status == "running":
                bubble.wait_terminal = "Waiting 0s"
            elif result_status == "completed":
                bubble.wait_terminal = "Command completed"
            elif result_status == "cancelled":
                bubble.wait_terminal = "Command cancelled"
            elif result_status == "timed_out":
                bubble.wait_terminal = "Command timed out"
            else:
                bubble.wait_terminal = status
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
            task
            for bubble in bubbles
            for task in (bubble.projection_task, bubble.wait_task)
            if task is not None and not task.done()
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
