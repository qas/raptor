"""Owned live console command and its provider-neutral presentation."""

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any

from raptor.state import session
from chat_format import bash_console_block
from chat_provider import (
    ActionButton,
    ConversationControlsProvider,
    Controls,
    ConversationId,
    IncomingAction,
    MessageId,
    ProcessOutputChunk,
)
from chat_runtime import detached_delivery_context, get_chat_provider
from config import CHAT_STREAM_INTERVAL
from observability import log_exception
from shell_sessions import (
    cancel_shell_session,
    run_shell,
    wait_shell_session,
)

_ACTION_PREFIX = "console-follow"
_SCREEN_ROWS = 24
_SCREEN_COLUMNS = 80
_MAX_COMMAND_CHARS = 900


class _TerminalScreen:
    """Bounded VT-style screen for PTY output projected into chat."""

    def __init__(
        self,
        rows: int = _SCREEN_ROWS,
        columns: int = _SCREEN_COLUMNS,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self._cells = [[" "] * columns for _ in range(rows)]
        self._row = 0
        self._column = 0
        self._pending = ""

    def _line_feed(self) -> None:
        if self._row < self.rows - 1:
            self._row += 1
            return
        self._cells.pop(0)
        self._cells.append([" "] * self.columns)

    def _write(self, character: str) -> None:
        if self._column >= self.columns:
            self._column = 0
            self._line_feed()
        self._cells[self._row][self._column] = character
        self._column += 1

    @staticmethod
    def _parameters(body: str) -> list[int]:
        body = body.lstrip("?><!")
        values: list[int] = []
        for value in body.split(";"):
            value = value.split(":", 1)[0]
            try:
                values.append(int(value) if value else 0)
            except ValueError:
                values.append(0)
        return values or [0]

    def _erase_display(self, mode: int) -> None:
        if mode in {2, 3}:
            self._cells = [
                [" "] * self.columns for _ in range(self.rows)
            ]
            return
        if mode == 1:
            for row in range(self._row):
                self._cells[row] = [" "] * self.columns
            end = min(self.columns, self._column + 1)
            self._cells[self._row][:end] = [" "] * end
            return
        self._cells[self._row][self._column:] = [" "] * (
            self.columns - self._column
        )
        for row in range(self._row + 1, self.rows):
            self._cells[row] = [" "] * self.columns

    def _erase_line(self, mode: int) -> None:
        if mode == 1:
            end = min(self.columns, self._column + 1)
            self._cells[self._row][:end] = [" "] * end
        elif mode == 2:
            self._cells[self._row] = [" "] * self.columns
        else:
            self._cells[self._row][self._column:] = [" "] * (
                self.columns - self._column
            )

    def _control_sequence(self, body: str, final: str) -> None:
        values = self._parameters(body)
        first = values[0]
        if final in {"H", "f"}:
            row = first or 1
            column = values[1] if len(values) > 1 else 1
            self._row = min(self.rows - 1, max(0, row - 1))
            self._column = min(self.columns - 1, max(0, (column or 1) - 1))
        elif final == "A":
            self._row = max(0, self._row - (first or 1))
        elif final == "B":
            self._row = min(self.rows - 1, self._row + (first or 1))
        elif final == "C":
            self._column = min(
                self.columns - 1,
                self._column + (first or 1),
            )
        elif final == "D":
            self._column = max(0, self._column - (first or 1))
        elif final == "G":
            self._column = min(self.columns - 1, max(0, (first or 1) - 1))
        elif final == "d":
            self._row = min(self.rows - 1, max(0, (first or 1) - 1))
        elif final == "J":
            self._erase_display(first)
        elif final == "K":
            self._erase_line(first)

    def feed(self, text: str) -> None:
        data = self._pending + text
        self._pending = ""
        index = 0
        while index < len(data):
            character = data[index]
            if character == "\x1b":
                if index + 1 >= len(data):
                    self._pending = data[index:]
                    return
                introducer = data[index + 1]
                if introducer == "[":
                    end = index + 2
                    while end < len(data) and not (
                        "@" <= data[end] <= "~"
                    ):
                        end += 1
                    if end >= len(data):
                        self._pending = data[index:]
                        return
                    self._control_sequence(
                        data[index + 2:end],
                        data[end],
                    )
                    index = end + 1
                    continue
                if introducer in {"(", ")"}:
                    if index + 2 >= len(data):
                        self._pending = data[index:]
                        return
                    index += 3
                    continue
                index += 2
                continue
            if character == "\r":
                self._column = 0
            elif character == "\n":
                self._line_feed()
            elif character == "\b":
                self._column = max(0, self._column - 1)
            elif character == "\t":
                self._column = min(
                    self.columns - 1,
                    ((self._column // 8) + 1) * 8,
                )
            elif character >= " ":
                self._write(character)
            index += 1

    def render(self) -> str:
        lines = ["".join(row).rstrip() for row in self._cells]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)


@dataclass
class _FollowConsole:
    token: str
    conversation_id: ConversationId
    command: str
    message_id: MessageId
    screen: _TerminalScreen = field(default_factory=_TerminalScreen)
    session_id: str | None = None
    task: asyncio.Task[None] | None = None
    projection_task: asyncio.Task[None] | None = None
    projection_pending: bool = False
    stop_requested: bool = False
    terminal: bool = False
    footer: str = ""

    def controls(self) -> Controls:
        if self.terminal or self.stop_requested:
            return ()
        return ((ActionButton("Stop", self.action),),)

    @property
    def action(self) -> str:
        return f"{_ACTION_PREFIX}:{self.token}:stop"

    def text(self) -> str:
        command = self.command[:_MAX_COMMAND_CHARS]
        if len(self.command) > _MAX_COMMAND_CHARS:
            command += "..."
        output = self.screen.render()
        if self.footer:
            output = output + ("\n" if output else "") + self.footer
        return bash_console_block(command, output)


_active: _FollowConsole | None = None


async def _project(console: _FollowConsole) -> None:
    try:
        await get_chat_provider().edit_message(
            console.conversation_id,
            console.message_id,
            console.text(),
            console.controls(),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception("console", "follow_projection_error", exc)


async def _drain_projection(console: _FollowConsole) -> None:
    first = True
    while console.projection_pending:
        if not first:
            await asyncio.sleep(CHAT_STREAM_INTERVAL)
        first = False
        console.projection_pending = False
        await _project(console)


def _queue_projection(console: _FollowConsole) -> None:
    console.projection_pending = True
    task = console.projection_task
    if task is None or task.done():
        console.projection_task = asyncio.create_task(
            _drain_projection(console)
        )


async def _flush_projection(console: _FollowConsole) -> None:
    task = console.projection_task
    if task is not None:
        await task


def _result_footer(result: dict[str, Any], *, stopped: bool) -> str:
    if stopped or result.get("status") == "cancelled":
        return "Process stopped."
    status = str(result.get("status") or "failed")
    error = str(result.get("error") or "").strip()
    exit_code = result.get("exit_code")
    if status == "completed":
        return "Process completed."
    if error:
        return error
    if isinstance(exit_code, int):
        return f"Process exited with code {exit_code}."
    return f"Process status: {status}."


async def _run_follow_console(console: _FollowConsole) -> None:
    global _active

    async def receive_output(chunk: ProcessOutputChunk) -> None:
        console.session_id = chunk.session_id
        console.screen.feed(chunk.text)
        if chunk.truncated:
            console.footer = "Output delivery was truncated."
        _queue_projection(console)

    result: dict[str, Any]
    try:
        result = await run_shell(
            console.command,
            timeout=0,
            yield_time_ms=250,
            tty=True,
            chat_id=console.conversation_id,
            parent_session_id=str(
                session.state.get("current_session_id") or ""
            ) or None,
            process_output=receive_output,
            max_published_output_chars=None,
            queue_completion_event=False,
        )
        session_id = str(result.get("session_id") or "")
        if session_id:
            console.session_id = session_id
        if result.get("status") == "running" and console.stop_requested:
            await cancel_shell_session(session_id)
        if result.get("status") == "running" and session_id:
            result = await wait_shell_session(session_id)
        console.footer = _result_footer(
            result,
            stopped=console.stop_requested,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception("console", "follow_command_error", exc)
        console.footer = f"Console error: {type(exc).__name__}: {exc}"
    finally:
        console.terminal = True
        console.projection_pending = False
        _queue_projection(console)
        await _flush_projection(console)
        if _active is console:
            _active = None


async def start_follow_console(
    conversation_id: ConversationId,
    command: str,
) -> str | None:
    """Start the single owned live console, returning an error if unavailable."""
    global _active
    provider = get_chat_provider()
    controls_supported = (
        provider.supports_controls(conversation_id)
        if isinstance(provider, ConversationControlsProvider)
        else provider.capabilities.controls
    )
    if not controls_supported:
        return "Follow mode requires interactive message controls."
    if _active is not None:
        return "A followed console command is already running."
    token = secrets.token_hex(6)
    controls = ((
        ActionButton("Stop", f"{_ACTION_PREFIX}:{token}:stop"),
    ),)
    message_id = await provider.create_message(
        conversation_id,
        bash_console_block(command[:_MAX_COMMAND_CHARS]),
        controls,
    )
    console = _FollowConsole(
        token=token,
        conversation_id=conversation_id,
        command=command,
        message_id=message_id,
    )
    _active = console
    with detached_delivery_context(conversation_id):
        console.task = asyncio.create_task(_run_follow_console(console))
    return None


async def _answer_action(
    action_id: str,
    text: str,
    *,
    alert: bool = False,
) -> None:
    provider = get_chat_provider()
    try:
        await provider.answer_action(action_id, text, alert=alert)
    except Exception as exc:
        log_exception(provider.name, "action_answer_error", exc)


async def handle_follow_console_action(action: IncomingAction) -> bool:
    parts = action.data.split(":")
    if len(parts) != 3 or parts[0] != _ACTION_PREFIX:
        return False
    provider = get_chat_provider()
    if action.sender_id != provider.authorized_user_id:
        await _answer_action(action.action_id, "Not authorized.", alert=True)
        return True
    token, decision = parts[1:]
    console = _active
    conversation_id = (
        action.presentation_conversation_id or action.conversation_id
    )
    if (
        decision != "stop"
        or console is None
        or console.token != token
        or console.conversation_id != conversation_id
        or console.message_id != action.message_id
        or console.terminal
    ):
        await _answer_action(
            action.action_id,
            "Console command is no longer running.",
        )
        return True
    await _answer_action(action.action_id, "Stopping")
    console.stop_requested = True
    console.footer = "Stopping..."
    _queue_projection(console)
    if console.session_id:
        await cancel_shell_session(console.session_id)
    return True


async def close_follow_console() -> None:
    """Stop and join the owned live console during application shutdown."""
    console = _active
    if console is None:
        return
    console.stop_requested = True
    if console.session_id:
        with session.bound_chat(console.conversation_id):
            await cancel_shell_session(console.session_id)
    if console.task is not None:
        await console.task
