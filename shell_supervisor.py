"""Own one shell command process group for the daemon's lifetime."""

import os
import select
import signal
import sys
import time

SUPERVISOR_MODE = "_shell-supervisor"


_POLL_SECONDS = 0.05
_TERMINATION_GRACE_SECONDS = 1.0


def _reap_if_exited(pid: int) -> int | None:
    waited, status = os.waitpid(pid, os.WNOHANG)
    if waited == 0:
        return None
    return status


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _terminate_group(pid: int, status: int | None) -> int:
    """Signal the group, then reap the leader if this process still owns it."""
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while status is None and time.monotonic() < deadline:
        status = _reap_if_exited(pid)
        if status is None:
            time.sleep(_POLL_SECONDS)
    _signal_group(pid, signal.SIGKILL)
    if status is None:
        _waited_pid, status = os.waitpid(pid, 0)
    return status


def _await_start(liveness_fd: int, start_fd: int) -> bool:
    while True:
        readable, _, _ = select.select(
            [liveness_fd, start_fd], [], [], _POLL_SECONDS
        )
        if liveness_fd in readable and not os.read(liveness_fd, 1):
            return False
        if start_fd in readable:
            return os.read(start_fd, 1) == b"1"


def _run(command: str, liveness_fd: int, start_fd: int) -> int:
    if not _await_start(liveness_fd, start_fd):
        return 128 + signal.SIGTERM
    os.close(start_fd)
    pid = os.fork()
    if pid == 0:
        os.setpgid(0, 0)
        os.close(liveness_fd)
        os.execl("/bin/bash", "bash", "-c", command)

    def forward_interrupt(_signum: int, _frame: object) -> None:
        _signal_group(pid, signal.SIGINT)

    signal.signal(signal.SIGINT, forward_interrupt)
    parent_lost = False
    status = None
    while True:
        readable, _, _ = select.select([liveness_fd], [], [], _POLL_SECONDS)
        if readable and not os.read(liveness_fd, 1):
            parent_lost = True
            break
        status = _reap_if_exited(pid)
        if status is not None:
            break
    status = _terminate_group(pid, status)
    if parent_lost:
        return 128 + signal.SIGTERM
    return os.waitstatus_to_exitcode(status)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    if len(args) != 4:
        return 2
    liveness_fd = int(args[1])
    start_fd = int(args[2])
    command = args[3]
    try:
        return _run(command, liveness_fd, start_fd)
    finally:
        os.close(liveness_fd)


if __name__ == "__main__":
    raise SystemExit(main())
