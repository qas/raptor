"""Runtime metadata and daemon CLI."""
import argparse
import json
import os
import select
import signal
import sys
import time
from typing import Any

from config_document import config_section, load_config_document
from process_lock import (
    detach_runtime_lock,
    refresh_runtime_lock_owner,
    release_runtime_lock,
    runtime_lock_status,
)
from runtime_paths import AGENT_WORKDIR, LOG_PATH, STATE_PATH
from storage import (
    FileTooLargeError,
    ensure_private_directory,
    read_bytes_bounded,
)
# ---------------------------------------------------------------------------
# Runtime / daemon metadata
# ---------------------------------------------------------------------------


def _runtime_state_load_limit() -> int:
    """Honor the state bound without importing application configuration."""
    configured = config_section(
        load_config_document(),
        "state",
        {"max_load_bytes"},
    ).get("max_load_bytes", 16_777_216)
    raw: object = os.environ.get("MAX_STATE_LOAD_BYTES", configured)
    try:
        if isinstance(raw, bool):
            raise ValueError
        return max(1024, int(raw))
    except (TypeError, ValueError):
        return 16_777_216


def runtime_info() -> dict[str, Any]:
    try:
        encoded = read_bytes_bounded(STATE_PATH, _runtime_state_load_limit())
        persisted = json.loads(encoded)
    except (
        OSError,
        UnicodeError,
        FileTooLargeError,
        json.JSONDecodeError,
    ):
        return {}
    if not isinstance(persisted, dict):
        return {}
    info = persisted.get("runtime")
    if isinstance(info, dict):
        return info
    return {}


def set_runtime(*, daemon: bool) -> None:
    from session import save_state, state

    refresh_runtime_lock_owner()
    state["runtime"] = {
        "pid": os.getpid(),
        "started_at": int(time.time()),
        "daemon": daemon,
        "log": str(LOG_PATH) if daemon else None,
    }
    save_state()


def clear_runtime_if_ours() -> None:
    from session import save_state, state

    info = state.get("runtime")

    if isinstance(info, dict) and info.get("pid") == os.getpid():
        state["runtime"] = {}
        save_state()


def runtime_uptime(info: dict[str, Any] | None = None) -> int:
    info = runtime_info() if info is None else info
    started = info.get("started_at")
    if not started:
        return 0
    try:
        return max(0, int(time.time() - int(started)))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Daemon CLI
# ---------------------------------------------------------------------------

DAEMON_START_TIMEOUT_SECONDS = 60.0


def _read_with_deadline(read_fd: int, deadline: float) -> bytes | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    readable, _writable, _exceptional = select.select(
        [read_fd],
        [],
        [],
        remaining,
    )
    if not readable:
        return None
    return os.read(read_fd, 1)


def _read_daemon_pid(read_fd: int, deadline: float) -> int | None:
    payload = bytearray()
    while len(payload) < 32:
        chunk = _read_with_deadline(read_fd, deadline)
        if not chunk:
            return None
        if chunk == b"\n":
            break
        payload.extend(chunk)
    try:
        pid = int(payload)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _await_daemon_ready(read_fd: int) -> tuple[bool, int | None]:
    deadline = time.monotonic() + DAEMON_START_TIMEOUT_SECONDS
    pid = _read_daemon_pid(read_fd, deadline)
    if pid is None:
        return False, None
    return _read_with_deadline(read_fd, deadline) == b"1", pid


def signal_daemon_ready(write_fd: int) -> None:
    """Release the launcher only after application initialization succeeds."""
    try:
        os.write(write_fd, b"1")
    finally:
        os.close(write_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raptor provider-neutral Responses API agent"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="show the Raptor version and exit",
    )
    mode.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="run detached in the background",
    )

    mode.add_argument(
        "--stop-daemon",
        action="store_true",
        help="stop the process that owns this Raptor runtime",
    )

    mode.add_argument(
        "--status",
        action="store_true",
        help="show local runtime status and exit",
    )

    mode.add_argument(
        "--check-proxy",
        action="store_true",
        help="test the configured proxy and show its public egress IP",
    )

    mode.add_argument(
        "--check-sandbox",
        action="store_true",
        help="test Linux Bubblewrap from Raptor's security context",
    )

    return parser.parse_args()


def cli_runtime_status() -> int:
    lock = runtime_lock_status()
    if not lock.held:
        print("Raptor: not running")
        return 1
    info = runtime_info()
    pid = lock.owner_pid
    metadata_matches = pid is not None and str(info.get("pid")) == str(pid)
    daemon = (
        "yes" if info.get("daemon") else "no"
    ) if metadata_matches else "unknown"
    uptime = f"{runtime_uptime(info)}s" if metadata_matches else "unknown"
    log_path = (info.get("log") or "-") if metadata_matches else "-"
    print(
        "Raptor: running "
        f"pid={pid if pid is not None else 'unknown'} "
        f"daemon={daemon} "
        f"uptime={uptime} "
        f"log={log_path}"
    )

    return 0


def stop_daemon() -> int:
    lock = runtime_lock_status()
    if not lock.held:
        print("Raptor: not running")
        return 1
    pid = lock.owner_pid
    if pid is None:
        print("Raptor: runtime is locked but its owner PID is unavailable")
        return 2

    if pid == os.getpid():
        print("Refusing to stop the current process")
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("Raptor: process already gone")
        return 1

    print(f"Raptor: sent SIGTERM to pid {pid}")

    return 0


def daemonize() -> int:
    """Detach and return the application-readiness pipe in the daemon."""

    ensure_private_directory(LOG_PATH.parent)
    ready_read_fd, ready_write_fd = os.pipe()
    first_pid = os.fork()

    if first_pid > 0:
        os.close(ready_write_fd)
        try:
            ready, daemon_pid = _await_daemon_ready(ready_read_fd)
        finally:
            os.close(ready_read_fd)
        if not ready:
            if daemon_pid is not None:
                try:
                    os.kill(daemon_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            release_runtime_lock()
            print("Raptor daemon failed to start", file=sys.stderr)
            raise SystemExit(1)
        print(f"Raptor daemon starting; log: {LOG_PATH}")
        detach_runtime_lock()
        raise SystemExit(0)

    os.close(ready_read_fd)
    os.setsid()

    second_pid = os.fork()

    if second_pid > 0:
        os.close(ready_write_fd)
        os._exit(0)

    try:
        os.write(ready_write_fd, f"{os.getpid()}\n".encode("ascii"))
        os.chdir(str(AGENT_WORKDIR))
        os.umask(0o027)
        stdin_fd = os.open(os.devnull, os.O_RDONLY)
        log_fd = os.open(
            LOG_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        os.fchmod(log_fd, 0o600)

        os.dup2(stdin_fd, sys.stdin.fileno())
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        os.close(stdin_fd)
        os.close(log_fd)
        refresh_runtime_lock_owner()
    except BaseException:
        os.close(ready_write_fd)
        raise
    return ready_write_fd
