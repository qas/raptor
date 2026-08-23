"""Runtime metadata and daemon CLI."""
import argparse
import os
import signal
import sys
import time
from typing import Any

from config import AGENT_WORKDIR, LOG_PATH
from process_lock import (
    refresh_runtime_lock_owner,
    runtime_lock_held,
)
from session import save_state, state

# ---------------------------------------------------------------------------
# Runtime / daemon metadata
# ---------------------------------------------------------------------------


def runtime_info() -> dict[str, Any]:
    info = state.get("runtime")
    if isinstance(info, dict):
        return info
    return {}


def clear_stale_runtime() -> None:
    info = runtime_info()

    pid = info.get("pid")
    if not pid:
        return
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        state["runtime"] = {}
        save_state()
        return
    if not runtime_lock_held():
        state["runtime"] = {}
        save_state()


def set_runtime(*, daemon: bool) -> None:
    refresh_runtime_lock_owner()
    state["runtime"] = {
        "pid": os.getpid(),
        "started_at": int(time.time()),
        "daemon": daemon,
        "log": str(LOG_PATH) if daemon else None,
    }
    save_state()


def clear_runtime_if_ours() -> None:
    info = runtime_info()

    if info.get("pid") == os.getpid():
        state["runtime"] = {}
        save_state()


def runtime_uptime() -> int:
    info = runtime_info()

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
    parser.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="run detached in the background",
    )

    parser.add_argument(
        "--stop-daemon",
        action="store_true",
        help="stop the process recorded in state runtime metadata",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="show local runtime status and exit",
    )

    return parser.parse_args()


def cli_runtime_status() -> int:
    clear_stale_runtime()

    info = runtime_info()

    pid = info.get("pid")
    if not pid:
        print("Raptor: not running")
        return 1
    print(
        "Raptor: running "
        f"pid={pid} "
        f"daemon="
        f"{'yes' if info.get('daemon') else 'no'} "
        f"uptime={runtime_uptime()}s "
        f"log={info.get('log') or '-'}"
    )

    return 0


def stop_daemon_from_state() -> int:
    clear_stale_runtime()

    info = runtime_info()

    pid = info.get("pid")
    if not pid or not runtime_lock_held():
        if pid:
            state["runtime"] = {}
            save_state()
        print("Raptor: no running process recorded")
        return 1

    pid = int(pid)

    if pid == os.getpid():
        print("Refusing to stop the current process")
        return 2

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        state["runtime"] = {}
        save_state()

        print("Raptor: process already gone")
        return 1

    print(f"Raptor: sent SIGTERM to pid {pid}")

    return 0


def daemonize() -> None:
    """Classic Unix double-fork daemonization."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    first_pid = os.fork()

    if first_pid > 0:
        print(f"Raptor daemon starting; log: {LOG_PATH}")
        raise SystemExit(0)

    os.setsid()

    second_pid = os.fork()

    if second_pid > 0:
        os._exit(0)

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
