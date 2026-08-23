"""Atomic process ownership independent of runtime state loading."""

import fcntl
import os

from config import RAPTOR_HOME


RUNTIME_LOCK_PATH = RAPTOR_HOME / "runtime.lock"
_runtime_lock_fd: int | None = None


def acquire_runtime_lock() -> bool:
    """Acquire the process-lifetime single-instance lock."""
    global _runtime_lock_fd
    if _runtime_lock_fd is not None:
        return True
    RUNTIME_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(RUNTIME_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return False
    _runtime_lock_fd = fd
    refresh_runtime_lock_owner()
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


def refresh_runtime_lock_owner() -> None:
    """Refresh the diagnostic PID after daemonization forks."""
    if _runtime_lock_fd is None:
        return
    os.lseek(_runtime_lock_fd, 0, os.SEEK_SET)
    os.ftruncate(_runtime_lock_fd, 0)
    os.write(_runtime_lock_fd, str(os.getpid()).encode("ascii"))


def runtime_lock_held() -> bool:
    """Return whether any process currently owns the runtime lock."""
    if _runtime_lock_fd is not None:
        return True
    RUNTIME_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(RUNTIME_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
