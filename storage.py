"""Crash-safe local file replacement primitives."""

import os
from pathlib import Path


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
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if mode is not None:
        os.chmod(temporary_path, mode)
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def write_text_atomic(
    path: Path,
    text: str,
    *,
    mode: int | None = None,
) -> None:
    write_bytes_atomic(path, text.encode("utf-8"), mode=mode)
