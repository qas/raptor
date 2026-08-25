"""Single-owner lifecycle for root turns."""

import asyncio
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Callable, Coroutine
from typing import Any

from observability import log_event


INTERRUPT_GRACE_SECONDS = 3.0


class TurnKind(StrEnum):
    REGULAR = "regular"
    MANUAL_COMPACTION = "manual_compaction"
    STATELESS_ASK = "stateless_ask"


@dataclass(frozen=True)
class TurnSnapshot:
    id: str
    kind: TurnKind
    started_at: float
    goal_id: str | None


@dataclass(frozen=True)
class InterruptResult:
    snapshot: TurnSnapshot | None
    completed: bool
    error: BaseException | None = None

    @property
    def interrupted(self) -> bool:
        return self.snapshot is not None


@dataclass
class _ActiveTurn:
    snapshot: TurnSnapshot
    task: asyncio.Task[None]


class TurnCoordinator:
    """Own exactly one root task and its lifecycle metadata."""

    def __init__(self) -> None:
        self._active: _ActiveTurn | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._active.task if self._active else None

    @property
    def snapshot(self) -> TurnSnapshot | None:
        return self._active.snapshot if self._active else None

    @property
    def goal_id(self) -> str | None:
        snapshot = self.snapshot
        return snapshot.goal_id if snapshot else None

    def is_running(self) -> bool:
        task = self.task
        return bool(task and not task.done())

    def elapsed_seconds(self) -> int:
        snapshot = self.snapshot
        if snapshot is None or not self.is_running():
            return 0
        return max(0, int(time.monotonic() - snapshot.started_at))

    def start(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        kind: TurnKind,
        goal_id: str | None = None,
        before_start: Callable[[TurnSnapshot], None] | None = None,
    ) -> asyncio.Task[None]:
        if self.is_running():
            coroutine.close()
            raise RuntimeError("a root turn is already running")
        snapshot = TurnSnapshot(
            id=secrets.token_hex(8),
            kind=kind,
            started_at=time.monotonic(),
            goal_id=goal_id,
        )
        try:
            if before_start is not None:
                before_start(snapshot)
        except Exception:
            coroutine.close()
            raise
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self._observe_task)
        self._active = _ActiveTurn(snapshot=snapshot, task=task)
        return task

    @staticmethod
    def _observe_task(task: asyncio.Task[None]) -> None:
        """Consume and report failures from detached root tasks."""
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        log_event(
            "agent",
            "root_task_error",
            {
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def set_goal_id(self, goal_id: str | None) -> None:
        active = self._active
        if active is None:
            return
        snapshot = active.snapshot
        active.snapshot = TurnSnapshot(
            id=snapshot.id,
            kind=snapshot.kind,
            started_at=snapshot.started_at,
            goal_id=goal_id,
        )

    def finish(self, task: asyncio.Task[Any] | None = None) -> bool:
        active = self._active
        if active is None:
            return False
        if task is not None and active.task is not task:
            return False
        self._active = None
        return True

    async def interrupt(
        self,
        *,
        expected_turn_id: str | None = None,
        grace_seconds: float = INTERRUPT_GRACE_SECONDS,
    ) -> InterruptResult:
        active = self._active
        if active is None or active.task.done():
            return InterruptResult(snapshot=None, completed=True)
        if (
            expected_turn_id is not None
            and active.snapshot.id != expected_turn_id
        ):
            return InterruptResult(snapshot=None, completed=False)

        active.task.cancel()
        done, _pending = await asyncio.wait(
            {active.task},
            timeout=max(0.0, grace_seconds),
        )
        if not done:
            active.task.cancel()
            return InterruptResult(snapshot=active.snapshot, completed=False)
        try:
            active.task.result()
        except asyncio.CancelledError:
            return InterruptResult(snapshot=active.snapshot, completed=True)
        except BaseException as exc:
            return InterruptResult(
                snapshot=active.snapshot,
                completed=True,
                error=exc,
            )
        return InterruptResult(snapshot=active.snapshot, completed=True)


class TurnRegistry:
    """Route turn ownership to the currently bound main-agent chat."""

    @staticmethod
    def _current() -> TurnCoordinator:
        from session import current_runtime

        return current_runtime().turns

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._current().task

    @property
    def snapshot(self) -> TurnSnapshot | None:
        return self._current().snapshot

    @property
    def goal_id(self) -> str | None:
        return self._current().goal_id

    def is_running(self) -> bool:
        return self._current().is_running()

    def elapsed_seconds(self) -> int:
        return self._current().elapsed_seconds()

    def start(
        self,
        coroutine: Coroutine[Any, Any, None],
        *,
        kind: TurnKind,
        goal_id: str | None = None,
        before_start: Callable[[TurnSnapshot], None] | None = None,
    ) -> asyncio.Task[None]:
        return self._current().start(
            coroutine,
            kind=kind,
            goal_id=goal_id,
            before_start=before_start,
        )

    def set_goal_id(self, goal_id: str | None) -> None:
        self._current().set_goal_id(goal_id)

    def finish(self, task: asyncio.Task[Any] | None = None) -> bool:
        return self._current().finish(task)

    async def interrupt(
        self,
        *,
        expected_turn_id: str | None = None,
        grace_seconds: float = INTERRUPT_GRACE_SECONDS,
    ) -> InterruptResult:
        return await self._current().interrupt(
            expected_turn_id=expected_turn_id,
            grace_seconds=grace_seconds,
        )


turns = TurnRegistry()
