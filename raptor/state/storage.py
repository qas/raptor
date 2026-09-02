"""Crash-safe local file replacement primitives."""

import os
import tempfile
from pathlib import Path


class FileTooLargeError(ValueError):
    """A bounded file read exceeded its explicit acquisition limit."""


def read_bytes_bounded(path: Path, limit: int) -> bytes:
    """Read at most ``limit`` bytes or reject the file as oversized."""
    if limit < 0:
        raise ValueError("read limit must not be negative")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise FileTooLargeError(f"file exceeds {limit} bytes")
    return data


def read_text_bounded(
    path: Path,
    limit: int,
    *,
    errors: str = "strict",
) -> str:
    """Read bounded UTF-8 text without a check-then-read race."""
    return read_bytes_bounded(path, limit).decode("utf-8", errors=errors)


def ensure_private_directory(path: Path) -> None:
    """Create a process-state directory and enforce owner-only access."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def fsync_directory(directory: Path) -> None:
    """Durably commit directory entries after file creation or replacement."""
    fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_atomic(
    path: Path,
    data: bytes,
    *,
    mode: int | None = None,
) -> None:
    """Replace a file only after its new contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    fsync_directory(path.parent)


def write_bytes_exclusive_atomic(
    path: Path,
    data: bytes,
    *,
    mode: int | None = None,
) -> None:
    """Publish a complete new file atomically without replacing a target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.link(temporary_path, path, follow_symlinks=False)
        linked = True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    if linked:
        fsync_directory(path.parent)


def write_text_atomic(
    path: Path,
    text: str,
    *,
    mode: int | None = None,
) -> None:
    write_bytes_atomic(path, text.encode("utf-8"), mode=mode)
