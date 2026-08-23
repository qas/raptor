"""Runtime metadata and daemon CLI."""
import argparse
import json
import os
import signal
import sys
import time
from typing import Any

from process_lock import (
    detach_runtime_lock,
    refresh_runtime_lock_owner,
    release_runtime_lock,
    runtime_lock_status,
)
from runtime_paths import AGENT_WORKDIR, LOG_PATH, STATE_PATH

# ---------------------------------------------------------------------------
# Runtime / daemon metadata
# ---------------------------------------------------------------------------


def runtime_info() -> dict[str, Any]:
    try:
        persisted = json.loads(STATE_PATH.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raptor provider-neutral Responses API agent"
    )
    mode = parser.add_mutually_exclusive_group()
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


def daemonize() -> None:
    """Classic Unix double-fork daemonization."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ready_read_fd, ready_write_fd = os.pipe()
    first_pid = os.fork()

    if first_pid > 0:
        os.close(ready_write_fd)
        try:
            ready = os.read(ready_read_fd, 1)
        finally:
            os.close(ready_read_fd)
        if ready != b"1":
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
        os.chdir(str(AGENT_WORKDIR))
        os.umask(0o027)
        stdin_fd = os.open(os.devnull, os.O_RDONLY)
        log_fd = os.open(
            LOG_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )

        os.dup2(stdin_fd, sys.stdin.fileno())
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        os.close(stdin_fd)
        os.close(log_fd)
        refresh_runtime_lock_owner()
        os.write(ready_write_fd, b"1")
    finally:
        os.close(ready_write_fd)
