"""Managed shell sessions with yielding and completion events."""

import asyncio
import codecs
import errno
import json
import os
import pty
import secrets
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import session as runtime_session
from chat_provider import ConversationId
from config import AGENT_WORKDIR, FILESYSTEM_POLICY, MAX_TOOL_OUTPUT, SHELL_TIMEOUT
from observability import log_event, log_shell_start
from shell_supervisor import SUPERVISOR_MODE

MAX_YIELD_TIME_MS = 30_000
MAX_POLL_TIME_MS = 300_000
MAX_WRITE_TIME_MS = 30_000
MIN_YIELD_TIME_MS = 250
MIN_POLL_TIME_MS = 5_000
DEFAULT_YIELD_TIME_MS = 10_000
DEFAULT_POLL_TIME_MS = 5_000
DEFAULT_WRITE_TIME_MS = 250
# The supervisor gives its child group one second to exit before SIGKILL.
# The owner must wait longer so it never kills the supervisor mid-cleanup.
TERMINATION_GRACE_SECONDS = 2.0
MAX_RETAINED_SESSIONS = 50
MAX_LIVE_SESSIONS = 64
_SUPERVISOR_PATH = Path(__file__).with_name("shell_supervisor.py")
_SUPERVISOR_EXECUTABLE = os.path.realpath(sys.executable)


def supervisor_argv() -> list[str]:
    """Launch the supervisor through this process's original executable."""
    if getattr(sys, "frozen", False):
        return [_SUPERVISOR_EXECUTABLE, SUPERVISOR_MODE]
    return [_SUPERVISOR_EXECUTABLE, str(_SUPERVISOR_PATH)]


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
    chat_key: str
    parent_session_id: str | None
    process: asyncio.subprocess.Process
    timeout: int | None
    tty: bool = False
    pty_write_fd: int | None = None
    liveness_write_fd: int | None = None
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
_spawning_sessions = 0


def _owned_sessions() -> list[ShellSession]:
    chat_key = runtime_session.current_runtime().key
    return [item for item in _sessions.values() if item.chat_key == chat_key]


def _require_owned_session(session_id: str) -> ShellSession | None:
    item = _sessions.get(session_id)
    if item is None or item.chat_key != runtime_session.current_runtime().key:
        return None
    return item


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
    _close_liveness_guard(session)
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=TERMINATION_GRACE_SECONDS,
        )
    except asyncio.TimeoutError:
        _signal_process_group(process, signal.SIGKILL)
        await process.wait()


def _close_liveness_guard(session: ShellSession) -> None:
    fd = session.liveness_write_fd
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass
    session.liveness_write_fd = None


def _release_start_gate(fd: int) -> None:
    os.write(fd, b"1")


async def _abort_unstarted_shell(
    shell_session: ShellSession,
    start_fd: int,
    *,
    stage: str,
    error: BaseException,
) -> dict[str, Any]:
    try:
        os.close(start_fd)
    except OSError:
        pass
    shell_session.detached = False
    shell_session.initial_decided.set()
    await _terminate(shell_session)
    await shell_session.done.wait()
    _sessions.pop(shell_session.id, None)
    return {
        "ok": False,
        "error": f"shell {stage} failed: {type(error).__name__}",
    }


async def _monitor(
    session: ShellSession,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
) -> None:
    try:
        try:
            if session.timeout is None:
                await session.process.wait()
            else:
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
        _close_liveness_guard(session)
        session.completed_at = time.time()
        session.done.set()
        await session.initial_decided.wait()
        if (
            session.detached
            and session.chat_id != ""
            and session.status != "cancelled"
        ):
            session.completion_pending = True
            _queue_shell_completion(session.id)
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


def _truncate_output(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n... [truncated] ...\n"
    if limit <= len(marker):
        return text[:limit], True
    remaining = limit - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:], True


def _fit_output_pair(stdout: str, stderr: str) -> tuple[str, str, bool]:
    """Fit both streams within the single tool-output budget."""
    if len(stdout) + len(stderr) <= MAX_TOOL_OUTPUT:
        return stdout, stderr, False
    stdout_limit = min(len(stdout), MAX_TOOL_OUTPUT // 2)
    stderr_limit = min(len(stderr), MAX_TOOL_OUTPUT // 2)
    remaining = MAX_TOOL_OUTPUT - stdout_limit - stderr_limit
    stdout_extra = min(remaining, len(stdout) - stdout_limit)
    stdout_limit += stdout_extra
    remaining -= stdout_extra
    stderr_limit += min(remaining, len(stderr) - stderr_limit)
    fitted_stdout, stdout_truncated = _truncate_output(stdout, stdout_limit)
    fitted_stderr, stderr_truncated = _truncate_output(stderr, stderr_limit)
    return (
        fitted_stdout,
        fitted_stderr,
        stdout_truncated or stderr_truncated,
    )


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
    stdout, stderr, pair_truncated = _fit_output_pair(stdout, stderr)
    truncated = truncated or pair_truncated
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
            if item.status != "running" and not item.completion_pending
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
    runtime = runtime_session.current_runtime()
    if (
        chat_id is not None
        and runtime_session.conversation_key(chat_id) != runtime.key
    ):
        return {
            "ok": False,
            "error": "shell conversation does not match the current chat",
        }
    if len(command) > MAX_TOOL_OUTPUT:
        return {
            "ok": False,
            "error": f"shell command exceeds {MAX_TOOL_OUTPUT} characters",
        }
    requested_timeout = SHELL_TIMEOUT if timeout is None else int(timeout)
    if requested_timeout < 0:
        return {
            "ok": False,
            "error": "shell timeout must be zero or greater",
        }
    timeout = requested_timeout or None
    yield_ms = min(
        MAX_YIELD_TIME_MS,
        max(MIN_YIELD_TIME_MS, int(
            DEFAULT_YIELD_TIME_MS
            if yield_time_ms is None
            else yield_time_ms
        )),
    )
    global _spawning_sessions
    global_running = sum(
        item.status == "running" for item in _sessions.values()
    )
    if global_running + _spawning_sessions >= MAX_LIVE_SESSIONS:
        return {
            "ok": False,
            "error": f"too many live shell sessions (limit {MAX_LIVE_SESSIONS})",
        }
    pending_completions = sum(
        item.completion_pending for item in _sessions.values()
    )
    if pending_completions >= MAX_RETAINED_SESSIONS:
        return {
            "ok": False,
            "error": (
                "pending shell completions must be delivered before more "
                "shell sessions can start"
            ),
        }
    _spawning_sessions += 1
    pty_write_fd: int | None = None
    liveness_read_fd: int | None = None
    liveness_write_fd: int | None = None
    start_read_fd: int | None = None
    start_write_fd: int | None = None
    policy_file = None
    try:
        policy_file = tempfile.TemporaryFile(mode="w+b")
        policy_file.write(FILESYSTEM_POLICY.shell_payload().encode("utf-8"))
        policy_file.flush()
        policy_file.seek(0)
        policy_fd = policy_file.fileno()
        liveness_read_fd, liveness_write_fd = os.pipe()
        start_read_fd, start_write_fd = os.pipe()
        if tty:
            master_fd, slave_fd = pty.openpty()
            try:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *supervisor_argv(),
                        str(liveness_read_fd),
                        str(start_read_fd),
                        str(policy_fd),
                        command,
                        cwd=str(AGENT_WORKDIR),
                        stdin=slave_fd,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        start_new_session=True,
                        pass_fds=(liveness_read_fd, start_read_fd, policy_fd),
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
                *supervisor_argv(),
                str(liveness_read_fd),
                str(start_read_fd),
                str(policy_fd),
                command,
                cwd=str(AGENT_WORKDIR),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                pass_fds=(liveness_read_fd, start_read_fd, policy_fd),
            )
            stdout_stream = process.stdout
            stderr_stream = process.stderr
    except BaseException:
        if liveness_write_fd is not None:
            os.close(liveness_write_fd)
        if start_write_fd is not None:
            os.close(start_write_fd)
        raise
    finally:
        if liveness_read_fd is not None:
            os.close(liveness_read_fd)
        if start_read_fd is not None:
            os.close(start_read_fd)
        if policy_file is not None:
            policy_file.close()
        _spawning_sessions -= 1
    shell_session = ShellSession(
        id=_new_session_id(),
        command=command,
        chat_id=chat_id if chat_id is not None else "",
        chat_key=runtime.key,
        parent_session_id=parent_session_id,
        process=process,
        timeout=timeout,
        tty=tty,
        pty_write_fd=pty_write_fd,
        liveness_write_fd=liveness_write_fd,
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
        log_shell_start(
            session_id=shell_session.id,
            command=command,
            chat_id=shell_session.chat_id,
            parent_session_id=parent_session_id,
            pid=process.pid,
            timeout=timeout,
            tty=tty,
        )
    except Exception as exc:
        return await _abort_unstarted_shell(
            shell_session,
            start_write_fd,
            stage="audit",
            error=exc,
        )
    try:
        _release_start_gate(start_write_fd)
    except OSError as exc:
        return await _abort_unstarted_shell(
            shell_session,
            start_write_fd,
            stage="start",
            error=exc,
        )
    else:
        os.close(start_write_fd)
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
    shell_session = _require_owned_session(session_id)
    if shell_session is None:
        return {"ok": False, "error": f"unknown shell session: {session_id}"}
    chars = str(args.get("chars") or "")
    if len(chars) > MAX_TOOL_OUTPUT:
        return {
            "ok": False,
            "error": f"shell input exceeds {MAX_TOOL_OUTPUT} characters",
        }
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
    return sum(item.status == "running" for item in _owned_sessions())


def pending_shell_completions() -> int:
    return sum(item.completion_pending for item in _owned_sessions())


async def requeue_deferred_shell_completions() -> int:
    """Retry deferred completion delivery after explicit user activity."""
    items = [
        item
        for item in _owned_sessions()
        if item.completion_pending and item.completion_attempts > 0
    ]
    for item in items:
        item.completion_attempts = 0
        _queue_shell_completion(item.id)
    return len(items)


def _new_session_id() -> str:
    while True:
        session_id = secrets.token_hex(4)
        if session_id not in _sessions:
            return session_id


async def cancel_shell_sessions() -> int:
    owned = _owned_sessions()
    active = [item for item in owned if item.status == "running"]
    for item in owned:
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


async def cancel_shell_session(session_id: str) -> dict[str, Any]:
    """Cancel one live managed shell process group."""
    item = _require_owned_session(session_id)
    if item is None:
        return {
            "ok": False,
            "kind": "shell",
            "id": session_id,
            "error": f"unknown shell session: {session_id}",
        }
    async with item.interaction_lock:
        if item.status != "running" or item.process.returncode is not None:
            if item.monitor_task is not None:
                await asyncio.gather(item.monitor_task, return_exceptions=True)
            return {
                "ok": False,
                "kind": "shell",
                "id": session_id,
                "status": item.status,
                "error": "shell session is not running",
            }
        item.detached = False
        item.completion_pending = False
        item.completion_attempts = 0
        item.initial_decided.set()
        item.status = "cancelled"
        item.error = "command was cancelled"
        await _terminate(item)
        if item.monitor_task is not None:
            await asyncio.gather(item.monitor_task, return_exceptions=True)
    return {
        "ok": True,
        "kind": "shell",
        "id": session_id,
        "status": item.status,
        "exit_code": item.exit_code,
    }


def _completion_prompt(session: ShellSession) -> str:
    result = _result(session, drain=False)
    payload = {
        "session_id": session.id,
        "command": session.command,
        "status": session.status,
        "exit_code": session.exit_code,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "truncated": result["truncated"],
        "error": session.error,
    }
    return (
        "A background shell command has finished. Assess the result and send "
        "the user only the relevant outcome. Do not describe this notification "
        "as a new user request.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _queue_shell_completion(session_id: str) -> bool:
    from controller import enqueue_runtime_event
    from runtime_events import RuntimeEventKind

    item = _sessions.get(session_id)
    if item is None or not item.completion_pending:
        return False
    with runtime_session.bound_chat(item.chat_id):
        current_session_id = runtime_session.state.get("current_session_id")
        if (
            item.parent_session_id is not None
            and str(current_session_id) != item.parent_session_id
        ):
            item.completion_pending = False
            return False
        try:
            completion = enqueue_runtime_event(
                item.chat_id,
                RuntimeEventKind.SHELL_COMPLETED,
                _completion_prompt(item),
                is_active=lambda: item.completion_pending,
            )
        except Exception as exc:
            log_event(
                "shell",
                "completion_delivery_error",
                {
                    "session_id": item.id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            item.completion_attempts = 1
            return False

    def completed(done: asyncio.Future[bool]) -> None:
        current = _sessions.get(session_id)
        if current is None or not current.completion_pending:
            return
        try:
            delivered = not done.cancelled() and bool(done.result())
        except Exception as exc:
            delivered = False
            log_event(
                "shell",
                "completion_delivery_error",
                {
                    "session_id": session_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        if delivered:
            current.completion_pending = False
            current.completion_attempts = 0
            log_event(
                "shell",
                "completion_delivered",
                {"session_id": session_id},
            )
        else:
            current.completion_attempts += 1
            log_event(
                "shell",
                "completion_deferred",
                {
                    "session_id": session_id,
                    "attempts": current.completion_attempts,
                },
            )

    completion.add_done_callback(completed)
    return True


async def reset_shell_sessions_for_tests() -> None:
    for item in tuple(_sessions.values()):
        with runtime_session.bound_chat(item.chat_id):
            await cancel_shell_sessions()
    _sessions.clear()
