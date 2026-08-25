"""Own one shell command process group for the daemon's lifetime."""

import os
import select
import signal
import sys
import time


_POLL_SECONDS = 0.05
_TERMINATION_GRACE_SECONDS = 1.0


def _leader_exited(pid: int) -> bool:
    return os.waitid(
        os.P_PID,
        pid,
        os.WEXITED | os.WNOHANG | os.WNOWAIT,
    ) is not None


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _terminate_group(pid: int) -> int:
    """Terminate descendants while retaining the leader against PID reuse."""
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline and not _leader_exited(pid):
        time.sleep(_POLL_SECONDS)
    _signal_group(pid, signal.SIGKILL)
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
    while True:
        readable, _, _ = select.select([liveness_fd], [], [], _POLL_SECONDS)
        if readable and not os.read(liveness_fd, 1):
            parent_lost = True
            break
        if _leader_exited(pid):
            break

    status = _terminate_group(pid)
    if parent_lost:
        return 128 + signal.SIGTERM
    return os.waitstatus_to_exitcode(status)


def main() -> int:
    if len(sys.argv) != 4:
        return 2
    liveness_fd = int(sys.argv[1])
    start_fd = int(sys.argv[2])
    command = sys.argv[3]
    try:
        return _run(command, liveness_fd, start_fd)
    finally:
        os.close(liveness_fd)


if __name__ == "__main__":
    raise SystemExit(main())
