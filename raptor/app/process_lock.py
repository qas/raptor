"""Atomic process ownership independent of runtime state loading."""

import fcntl
import os
from dataclasses import dataclass

from runtime_paths import RAPTOR_HOME
from raptor.state.storage import ensure_private_directory


RUNTIME_LOCK_PATH = RAPTOR_HOME / "runtime.lock"
_LOCK_OWNER_BYTES = 64
_runtime_lock_fd: int | None = None


@dataclass(frozen=True)
class RuntimeLockStatus:
    held: bool
    owner_pid: int | None


def _lock_owner(fd: int) -> int | None:
    try:
        raw = os.pread(fd, _LOCK_OWNER_BYTES, 0).decode("ascii").strip()
        pid = int(raw)
    except (OSError, UnicodeError, ValueError):
        return None
    return pid if pid > 0 else None


def acquire_runtime_lock() -> bool:
    """Acquire the process-lifetime single-instance lock."""
    global _runtime_lock_fd
    if _runtime_lock_fd is not None:
        return True
    ensure_private_directory(RUNTIME_LOCK_PATH.parent)
    fd = os.open(RUNTIME_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.set_inheritable(fd, False)
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    except BaseException:
        os.close(fd)
        raise
    _runtime_lock_fd = fd
    try:
        refresh_runtime_lock_owner()
    except OSError:
        release_runtime_lock()
        raise
    return True


def release_runtime_lock() -> None:
    """Release this process's ownership lock, if held."""
    global _runtime_lock_fd
    fd = _runtime_lock_fd
    _runtime_lock_fd = None
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def detach_runtime_lock() -> None:
    """Close this fork parent's descriptor without unlocking the child."""
    global _runtime_lock_fd
    fd = _runtime_lock_fd
    _runtime_lock_fd = None
    if fd is not None:
        os.close(fd)


def refresh_runtime_lock_owner() -> None:
    """Refresh the diagnostic PID after daemonization forks."""
    if _runtime_lock_fd is None:
        return
    owner = str(os.getpid()).encode("ascii").ljust(_LOCK_OWNER_BYTES)
    written = os.pwrite(_runtime_lock_fd, owner, 0)
    if written != len(owner):
        raise OSError("Could not publish the runtime owner PID")


def runtime_lock_status() -> RuntimeLockStatus:
    """Inspect process ownership and its diagnostic PID atomically."""
    if _runtime_lock_fd is not None:
        return RuntimeLockStatus(True, _lock_owner(_runtime_lock_fd))
    ensure_private_directory(RUNTIME_LOCK_PATH.parent)
    fd = os.open(RUNTIME_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return RuntimeLockStatus(True, _lock_owner(fd))
        fcntl.flock(fd, fcntl.LOCK_UN)
        return RuntimeLockStatus(False, None)
    finally:
        os.close(fd)


def runtime_lock_held() -> bool:
    """Return whether any process currently owns the runtime lock."""
    return runtime_lock_status().held
