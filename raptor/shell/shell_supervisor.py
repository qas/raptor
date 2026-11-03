"""Own one shell command process group for the daemon's lifetime."""

import os
import select
import signal
import sys
import time
from collections.abc import Callable

from raptor.shell.filesystem_permissions import FileAccessPolicy
from raptor.shell.shell_sandbox import build_shell_sandbox_launch

SUPERVISOR_MODE = "_shell-supervisor"


_POLL_SECONDS = 0.05
_TERMINATION_GRACE_SECONDS = 1.0
_MAX_POLICY_BYTES = 256 * 1024
_MAX_COMMAND_BYTES = 256 * 1024


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


def _await_start(
    liveness_fd: int,
    start_fd: int,
    stopped: Callable[[], bool],
) -> bool:
    while not stopped():
        try:
            readable, _, _ = select.select(
                [liveness_fd, start_fd], [], [], _POLL_SECONDS
            )
        except InterruptedError:
            continue
        if liveness_fd in readable and not os.read(liveness_fd, 1):
            return False
        if start_fd in readable:
            return os.read(start_fd, 1) == b"1"
    return False


def _read_command(command_fd: int) -> str:
    with os.fdopen(command_fd, "rb") as command_file:
        encoded = command_file.read(_MAX_COMMAND_BYTES + 1)
    if len(encoded) > _MAX_COMMAND_BYTES:
        raise ValueError("shell command exceeds size limit")
    return encoded.decode("utf-8")


def _run(
    command: str,
    policy_fd: int,
    liveness_fd: int,
    start_fd: int,
    ready_fd: int,
) -> int:
    # The default SIGTERM action skips finally and leaves mode-000 deny-read
    # placeholders in the workspace. Exit only through the cleanup paths below.
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    if not _await_start(liveness_fd, start_fd, lambda: stop):
        return 128 + signal.SIGTERM
    os.close(start_fd)
    if stop:
        return 128 + signal.SIGTERM
    try:
        with os.fdopen(policy_fd, "rb") as policy_file:
            encoded_policy = policy_file.read(_MAX_POLICY_BYTES + 1)
        if len(encoded_policy) > _MAX_POLICY_BYTES:
            raise ValueError("shell filesystem policy exceeds size limit")
        policy_payload = encoded_policy.decode("utf-8")
        policy = FileAccessPolicy.from_supervisor_payload(policy_payload)
        launch = build_shell_sandbox_launch(command, policy)
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        print(f"raptor sandbox: {exc}", file=sys.stderr)
        return 126
    if stop:
        launch.cleanup()
        return 128 + signal.SIGTERM
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
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
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
        while not child_exited and not stop:
            try:
                readable, _, _ = select.select(
                    [liveness_fd], [], [], _POLL_SECONDS
                )
            except InterruptedError:
                continue
            if readable and not os.read(liveness_fd, 1):
                parent_lost = True
                break
        status = _terminate_group(pid, lambda: child_exited)
        if parent_lost or stop:
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
    command_fd = int(args[5])
    try:
        try:
            command = _read_command(command_fd)
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"raptor sandbox: {exc}", file=sys.stderr)
            return 126
        return _run(command, policy_fd, liveness_fd, start_fd, ready_fd)
    finally:
        os.close(liveness_fd)
        os.close(ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
