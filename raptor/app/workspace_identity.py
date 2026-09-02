"""Workspace-owned agent instructions and durable context."""

import os
import stat
from pathlib import Path

from raptor.runtime_paths import AGENT_WORKDIR
from raptor.state.storage import FileTooLargeError, fsync_directory


MAX_WORKSPACE_IDENTITY_FILE_BYTES = 32 * 1024

_TEMPLATES = {
    "AGENTS.md": """# Agent Instructions

<!-- Define this workspace's agent identity, conventions, and working style. -->
""",
    "MEMORY.md": """# Agent Memory

<!-- Record durable project context that should carry across sessions. -->
""",
}

_workspace_identity = ""


def _create_if_missing(path: Path, content: str) -> bool:
    """Create one template without racing or replacing operator content."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    fsync_directory(path.parent)
    return True


def _read_identity_text(path: Path) -> str:
    """Read one bounded regular file without following a replaced symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise RuntimeError(
                f"{path.name} must be a regular workspace file"
            ) from exc
        raise
    try:
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError(
                    f"{path.name} must be a regular workspace file"
                )
            data = handle.read(MAX_WORKSPACE_IDENTITY_FILE_BYTES + 1)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if len(data) > MAX_WORKSPACE_IDENTITY_FILE_BYTES:
        raise FileTooLargeError
    return data.decode("utf-8")


def initialize_workspace_identity(
    workdir: Path = AGENT_WORKDIR,
) -> tuple[str, ...]:
    """Create missing workspace files and cache their bounded contents."""
    global _workspace_identity

    workdir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    sections: list[str] = []
    for name, template in _TEMPLATES.items():
        path = workdir / name
        if _create_if_missing(path, template):
            created.append(name)
        try:
            content = _read_identity_text(path).strip()
        except FileTooLargeError as exc:
            raise RuntimeError(
                f"{name} exceeds {MAX_WORKSPACE_IDENTITY_FILE_BYTES} bytes"
            ) from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"{name} must be UTF-8 text") from exc
        if content:
            sections.append(f"WORKSPACE {name}:\n{content}")
    _workspace_identity = "\n\n".join(sections)
    return tuple(created)


def workspace_identity_instructions() -> str:
    """Return the workspace identity captured during process startup."""
    return _workspace_identity
