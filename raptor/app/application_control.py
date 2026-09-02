"""Explicit process-exit requests owned by the application task."""

from __future__ import annotations

import asyncio
from enum import Enum


class ExitRequest(str, Enum):
    SHUTDOWN = "shutdown"
    RESTART = "restart"


_application_task: asyncio.Task[None] | None = None
_exit_request: ExitRequest | None = None


def bind_application_task(task: asyncio.Task[None]) -> None:
    """Bind process controls to the one task that owns application cleanup."""
    global _application_task, _exit_request
    if _application_task is not None:
        raise RuntimeError("application control is already bound")
    _application_task = task
    _exit_request = None
    task.add_done_callback(_release_if_owned)


def unbind_application_task(task: asyncio.Task[None]) -> None:
    global _application_task
    if _application_task is task:
        _application_task = None


def _release_if_owned(task: asyncio.Future[None]) -> None:
    global _application_task
    if _application_task is task:
        _application_task = None


def application_control_available() -> bool:
    task = _application_task
    return task is not None and not task.done()


def request_application_exit(request: ExitRequest) -> bool:
    """Record an exit request for the event dispatcher to activate."""
    global _exit_request
    task = _application_task
    if task is None or task.done():
        return False
    _exit_request = request
    return True


def activate_application_exit() -> bool:
    """Cancel the application owner after transport finalization."""
    task = _application_task
    if _exit_request is None or task is None or task.done():
        return False
    task.cancel()
    return True


def current_exit_request() -> ExitRequest | None:
    return _exit_request


def discard_exit_request() -> None:
    """Abort an exit request whose admitting event could not be finalized."""
    global _exit_request
    _exit_request = None


def take_exit_request() -> ExitRequest | None:
    global _exit_request
    request = _exit_request
    _exit_request = None
    return request
