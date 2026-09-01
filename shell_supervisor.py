"""Own one shell command process group for the daemon's lifetime."""

import os
import select
import signal
import sys
import time
from collections.abc import Callable

from filesystem_permissions import FileAccessPolicy
from shell_sandbox import build_shell_sandbox_launch

SUPERVISOR_MODE = "_shell-supervisor"


_POLL_SECONDS = 0.05
_TERMINATION_GRACE_SECONDS = 1.0
_MAX_POLICY_BYTES = 256 * 1024


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except (PermissionError, ProcessLookupError):
        pass


def _terminate_group(pid: int, leader_exited: Callable[[], bool]) -> int:
    """Terminate descendants before reaping the process-group leader."""
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while not leader_exited() and time.monotonic() < deadline:
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


def _run(
    command: str,
    policy_fd: int,
    liveness_fd: int,
    start_fd: int,
    ready_fd: int,
) -> int:
    if not _await_start(liveness_fd, start_fd):
        return 128 + signal.SIGTERM
    os.close(start_fd)
    try:
        with os.fdopen(policy_fd, "rb") as policy_file:
            encoded_policy = policy_file.read(_MAX_POLICY_BYTES + 1)
        if len(encoded_policy) > _MAX_POLICY_BYTES:
            raise ValueError("shell filesystem policy exceeds size limit")
        policy_payload = encoded_policy.decode("utf-8")
        policy = FileAccessPolicy.from_supervisor_payload(policy_payload)
        launch = build_shell_sandbox_launch(command, policy)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"raptor sandbox: {exc}", file=sys.stderr)
        return 126
    child_exited = False

    def record_child_exit(_signum: int, _frame: object) -> None:
        nonlocal child_exited
        child_exited = True

    try:
        signal.signal(signal.SIGCHLD, record_child_exit)
        pid = os.fork()
    except BaseException:
        launch.cleanup()
        raise
    if pid == 0:
        os.setpgid(0, 0)
        os.close(liveness_fd)
        os.close(ready_fd)
        try:
            os.execv(launch.argv[0], launch.argv)
        except OSError as exc:
            print(f"raptor sandbox: {exc}", file=sys.stderr)
            os._exit(126)
    for fd in launch.inherited_fds:
        try:
            os.close(fd)
        except OSError:
            pass
    launch.inherited_fds.clear()
    try:
        os.write(ready_fd, b"1")
    except OSError:
        status = _terminate_group(pid, lambda: child_exited)
        launch.cleanup()
        return os.waitstatus_to_exitcode(status)

    def forward_interrupt(_signum: int, _frame: object) -> None:
        _signal_group(pid, signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, forward_interrupt)
        parent_lost = False
        while not child_exited:
            readable, _, _ = select.select(
                [liveness_fd], [], [], _POLL_SECONDS
            )
            if readable and not os.read(liveness_fd, 1):
                parent_lost = True
                break
        status = _terminate_group(pid, lambda: child_exited)
        if parent_lost:
            return 128 + signal.SIGTERM
        return os.waitstatus_to_exitcode(status)
    finally:
        launch.cleanup()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv if argv is None else argv
    if len(args) != 6:
        return 2
    liveness_fd = int(args[1])
    start_fd = int(args[2])
    policy_fd = int(args[3])
    ready_fd = int(args[4])
    command = args[5]
    try:
        return _run(command, policy_fd, liveness_fd, start_fd, ready_fd)
    finally:
        os.close(liveness_fd)
        os.close(ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
