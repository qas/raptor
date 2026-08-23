"""Managed shell sessions with yielding and completion events."""

import asyncio
import codecs
import errno
import json
import os
import pty
import secrets
import signal
import time
from dataclasses import dataclass, field
from typing import Any

from chat_provider import ConversationId
from config import AGENT_WORKDIR, MAX_TOOL_OUTPUT, SHELL_TIMEOUT
from observability import log_event


MAX_YIELD_TIME_MS = 30_000
MAX_POLL_TIME_MS = 300_000
MAX_WRITE_TIME_MS = 30_000
MIN_YIELD_TIME_MS = 250
MIN_POLL_TIME_MS = 5_000
DEFAULT_YIELD_TIME_MS = 10_000
DEFAULT_POLL_TIME_MS = 5_000
DEFAULT_WRITE_TIME_MS = 250
TERMINATION_GRACE_SECONDS = 1.0
MAX_RETAINED_SESSIONS = 50
MAX_LIVE_SESSIONS = 64


class HeadTailBuffer:
    """Bound output while preserving evidence from both ends."""

    def __init__(self, limit: int = MAX_TOOL_OUTPUT) -> None:
        self.limit = max(1000, limit)
        self.head = ""
        self.tail = ""
        self.omitted = 0

    def append(self, text: str) -> None:
        if not text:
            return
        keep_head = self.limit // 2
        keep_tail = self.limit - keep_head
        if not self.omitted:
            combined = self.head + text
            if len(combined) <= self.limit:
                self.head = combined
                return
            self.head = combined[:keep_head]
            self.tail = combined[-keep_tail:]
            self.omitted = len(combined) - self.limit
            return
        combined_tail = self.tail + text
        overflow = max(0, len(combined_tail) - keep_tail)
        self.omitted += overflow
        self.tail = combined_tail[-keep_tail:]

    def render(self, *, include_marker: bool = True) -> str:
        if not self.omitted:
            return self.head
        marker = (
            f"\n... [{self.omitted} characters omitted] ...\n"
            if include_marker
            else ""
        )
        return self.head + marker + self.tail


@dataclass
class ShellSession:
    id: str
    command: str
    chat_id: ConversationId
    parent_session_id: str | None
    process: asyncio.subprocess.Process
    timeout: int
    tty: bool = False
    pty_write_fd: int | None = None
    started_at: float = field(default_factory=time.time)
    status: str = "running"
    exit_code: int | None = None
    error: str | None = None
    completed_at: float | None = None
    detached: bool = False
    completion_pending: bool = False
    completion_attempts: int = 0
    stdout: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    stderr: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    pending_stdout: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    pending_stderr: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    initial_decided: asyncio.Event = field(default_factory=asyncio.Event)
    interaction_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    monitor_task: asyncio.Task[None] | None = None


_sessions: dict[str, ShellSession] = {}
_completion_events: asyncio.Queue[str] = asyncio.Queue()
_spawning_sessions = 0


def _append_output(
    session: ShellSession,
    stream_name: str,
    text: str,
) -> None:
    total = session.stdout if stream_name == "stdout" else session.stderr
    pending = (
        session.pending_stdout
        if stream_name == "stdout"
        else session.pending_stderr
    )
    total.append(text)
    pending.append(text)


async def _pump_stream(
    session: ShellSession,
    stream: asyncio.StreamReader | None,
    stream_name: str,
) -> None:
    if stream is None:
        return
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        try:
            chunk = await stream.read(8192)
        except OSError as exc:
            if session.tty and exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        _append_output(session, stream_name, decoder.decode(chunk))
    remainder = decoder.decode(b"", final=True)
    if remainder:
        _append_output(session, stream_name, remainder)


def _signal_process_group(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


async def _terminate(session: ShellSession) -> None:
    process = session.process
    if process.returncode is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=TERMINATION_GRACE_SECONDS,
        )
    except asyncio.TimeoutError:
        _signal_process_group(process, signal.SIGKILL)
        await process.wait()


async def _monitor(
    session: ShellSession,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    try:
        try:
            await asyncio.wait_for(
                session.process.wait(),
                timeout=session.timeout,
            )
        except asyncio.TimeoutError:
            session.status = "timed_out"
            session.error = f"command timed out after {session.timeout}s"
            await _terminate(session)
        await asyncio.gather(stdout_task, stderr_task)
        session.exit_code = session.process.returncode
        if session.status == "running":
            session.status = (
                "completed" if session.exit_code == 0 else "failed"
            )
    except asyncio.CancelledError:
        session.status = "cancelled"
        session.error = "command was cancelled"
        await _terminate(session)
        await asyncio.gather(
            stdout_task,
            stderr_task,
            return_exceptions=True,
        )
        session.exit_code = session.process.returncode
        raise
    except Exception as exc:
        session.status = "failed"
        session.error = f"{type(exc).__name__}: {exc}"
        await _terminate(session)
        await asyncio.gather(
            stdout_task,
            stderr_task,
            return_exceptions=True,
        )
        session.exit_code = session.process.returncode
    finally:
        if session.pty_write_fd is not None:
            try:
                os.close(session.pty_write_fd)
            except OSError:
                pass
            session.pty_write_fd = None
        session.completed_at = time.time()
        session.done.set()
        await session.initial_decided.wait()
        if (
            session.detached
            and session.chat_id != ""
            and session.status != "cancelled"
        ):
            session.completion_pending = True
            await _completion_events.put(session.id)
        log_event(
            "shell",
            "completed",
            {
                "session_id": session.id,
                "status": session.status,
                "exit_code": session.exit_code,
            },
        )


def _drain_pending(session: ShellSession) -> tuple[str, str, bool]:
    stdout = session.pending_stdout.render()
    stderr = session.pending_stderr.render()
    truncated = bool(
        session.pending_stdout.omitted
        or session.pending_stderr.omitted
    )
    session.pending_stdout = HeadTailBuffer()
    session.pending_stderr = HeadTailBuffer()
    return stdout, stderr, truncated


def _result(
    session: ShellSession,
    *,
    drain: bool,
) -> dict[str, Any]:
    if drain:
        stdout, stderr, truncated = _drain_pending(session)
    else:
        stdout = session.stdout.render()
        stderr = session.stderr.render()
        truncated = bool(session.stdout.omitted or session.stderr.omitted)
    return {
        "ok": session.status in {"running", "completed"},
        "session_id": session.id,
        "status": session.status,
        "exit_code": session.exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
        "error": session.error,
        "wall_time_seconds": round(
            (session.completed_at or time.time()) - session.started_at,
            3,
        ),
    }


def _prune_sessions() -> None:
    completed = sorted(
        (
            item
            for item in _sessions.values()
            if item.status != "running"
        ),
        key=lambda item: item.completed_at or item.started_at,
    )
    excess = max(0, len(_sessions) - MAX_RETAINED_SESSIONS)
    for item in completed[:excess]:
        _sessions.pop(item.id, None)


async def run_shell(
    command: str,
    *,
    timeout: int | None,
    yield_time_ms: int | None,
    tty: bool,
    chat_id: ConversationId | None,
    parent_session_id: str | None,
) -> dict[str, Any]:
    timeout = min(600, max(1, int(timeout or SHELL_TIMEOUT)))
    yield_ms = min(
        MAX_YIELD_TIME_MS,
        max(MIN_YIELD_TIME_MS, int(
            DEFAULT_YIELD_TIME_MS
            if yield_time_ms is None
            else yield_time_ms
        )),
    )
    global _spawning_sessions
    if running_shell_sessions() + _spawning_sessions >= MAX_LIVE_SESSIONS:
        return {
            "ok": False,
            "error": f"too many live shell sessions (limit {MAX_LIVE_SESSIONS})",
        }
    _spawning_sessions += 1
    pty_write_fd: int | None = None
    try:
        if tty:
            master_fd, slave_fd = pty.openpty()
            try:
                try:
                    process = await asyncio.create_subprocess_exec(
                        "/bin/bash",
                        "-c",
                        command,
                        cwd=str(AGENT_WORKDIR),
                        stdin=slave_fd,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        start_new_session=True,
                    )
                except BaseException:
                    os.close(master_fd)
                    raise
            finally:
                os.close(slave_fd)
            pty_write_fd = os.dup(master_fd)
            read_pipe = os.fdopen(master_fd, "rb", buffering=0)
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            try:
                await asyncio.get_running_loop().connect_read_pipe(
                    lambda: protocol,
                    read_pipe,
                )
            except BaseException:
                read_pipe.close()
                os.close(pty_write_fd)
                _signal_process_group(process, signal.SIGKILL)
                await process.wait()
                raise
            stdout_stream: asyncio.StreamReader | None = reader
            stderr_stream: asyncio.StreamReader | None = None
        else:
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                command,
                cwd=str(AGENT_WORKDIR),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout_stream = process.stdout
            stderr_stream = process.stderr
    finally:
        _spawning_sessions -= 1
    shell_session = ShellSession(
        id=_new_session_id(),
        command=command,
        chat_id=chat_id if chat_id is not None else "",
        parent_session_id=parent_session_id,
        process=process,
        timeout=timeout,
        tty=tty,
        pty_write_fd=pty_write_fd,
    )
    _sessions[shell_session.id] = shell_session
    stdout_task = asyncio.create_task(
        _pump_stream(shell_session, stdout_stream, "stdout")
    )
    stderr_task = asyncio.create_task(
        _pump_stream(shell_session, stderr_stream, "stderr")
    )
    shell_session.monitor_task = asyncio.create_task(
        _monitor(shell_session, stdout_task, stderr_task)
    )
    try:
        if yield_ms:
            await asyncio.wait_for(
                shell_session.done.wait(),
                timeout=yield_ms / 1000,
            )
    except asyncio.TimeoutError:
        pass
    finally:
        shell_session.detached = not shell_session.done.is_set()
        shell_session.initial_decided.set()
    _prune_sessions()
    result = _result(shell_session, drain=True)
    if not shell_session.detached:
        result["session_id"] = None
    return result


async def write_stdin(args: dict[str, Any]) -> dict[str, Any]:
    session_id = str(args.get("session_id") or "").strip()
    shell_session = _sessions.get(session_id)
    if shell_session is None:
        return {"ok": False, "error": f"unknown shell session: {session_id}"}
    chars = str(args.get("chars") or "")
    requested_yield = int(
        args.get("yield_time_ms")
        if args.get("yield_time_ms") is not None
        else (
            DEFAULT_WRITE_TIME_MS
            if chars
            else DEFAULT_POLL_TIME_MS
        )
    )
    yield_ms = (
        min(MAX_WRITE_TIME_MS, max(MIN_YIELD_TIME_MS, requested_yield))
        if chars
        else min(MAX_POLL_TIME_MS, max(MIN_POLL_TIME_MS, requested_yield))
    )
    async with shell_session.interaction_lock:
        if chars and not shell_session.done.is_set():
            if chars == "\x03":
                _signal_process_group(shell_session.process, signal.SIGINT)
            elif shell_session.pty_write_fd is not None:
                try:
                    os.write(shell_session.pty_write_fd, chars.encode())
                except OSError:
                    return {"ok": False, "error": "stdin is closed"}
            else:
                stdin = shell_session.process.stdin
                if stdin is None or stdin.is_closing():
                    return {"ok": False, "error": "stdin is closed"}
                try:
                    stdin.write(chars.encode())
                    await stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    return {"ok": False, "error": "stdin is closed"}
        if not shell_session.done.is_set() and yield_ms:
            try:
                await asyncio.wait_for(
                    shell_session.done.wait(),
                    timeout=yield_ms / 1000,
                )
            except asyncio.TimeoutError:
                pass
        result = _result(shell_session, drain=True)
        if shell_session.done.is_set():
            # The active model turn observed completion; suppress a duplicate
            # asynchronous completion turn if it has not been delivered yet.
            shell_session.completion_pending = False
        return result


def running_shell_sessions() -> int:
    return sum(item.status == "running" for item in _sessions.values())


def pending_shell_completions() -> int:
    return sum(item.completion_pending for item in _sessions.values())


async def requeue_deferred_shell_completions() -> int:
    """Retry deferred completion delivery after explicit user activity."""
    items = [
        item
        for item in _sessions.values()
        if item.completion_pending and item.completion_attempts > 0
    ]
    for item in items:
        item.completion_attempts = 0
        await _completion_events.put(item.id)
    return len(items)


def _new_session_id() -> str:
    while True:
        session_id = secrets.token_hex(4)
        if session_id not in _sessions:
            return session_id


async def cancel_shell_sessions() -> int:
    active = [item for item in _sessions.values() if item.status == "running"]
    for item in _sessions.values():
        item.completion_pending = False
    for item in active:
        item.detached = False
        item.initial_decided.set()
        item.status = "cancelled"
        item.error = "command was cancelled"
    if active:
        await asyncio.gather(
            *(_terminate(item) for item in active),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(item.monitor_task for item in active if item.monitor_task),
            return_exceptions=True,
        )
    return len(active)


def _completion_prompt(session: ShellSession) -> str:
    payload = {
        "session_id": session.id,
        "command": session.command,
        "status": session.status,
        "exit_code": session.exit_code,
        "stdout": session.stdout.render(),
        "stderr": session.stderr.render(),
        "error": session.error,
    }
    return (
        "A background shell command has finished. Assess the result and send "
        "the user only the relevant outcome. Do not describe this notification "
        "as a new user request.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


async def shell_completion_event_loop() -> None:
    from controller import enqueue_internal_input
    from session import state

    while True:
        session_id = await _completion_events.get()
        try:
            item = _sessions.get(session_id)
            if item is None or not item.completion_pending:
                continue
            current_session_id = state.get("current_session_id")
            if (
                item.parent_session_id is not None
                and str(current_session_id) != item.parent_session_id
            ):
                item.completion_pending = False
                continue
            try:
                delivered = await enqueue_internal_input(
                    item.chat_id,
                    _completion_prompt(item),
                    is_active=lambda: item.completion_pending,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                delivered = False
                log_event(
                    "shell",
                    "completion_delivery_error",
                    {
                        "session_id": item.id,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            if delivered:
                item.completion_pending = False
                item.completion_attempts = 0
            else:
                item.completion_attempts += 1
                log_event(
                    "shell",
                    "completion_deferred",
                    {
                        "session_id": item.id,
                        "attempts": item.completion_attempts,
                    },
                )
        finally:
            _completion_events.task_done()


async def reset_shell_sessions_for_tests() -> None:
    global _completion_events
    await cancel_shell_sessions()
    _sessions.clear()
    _completion_events = asyncio.Queue()
