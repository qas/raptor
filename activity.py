"""Provider-neutral projection of background subagent activity."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from chat_provider import ConversationId
from chat_runtime import get_chat_provider
from observability import log_exception
import session


ACTIVITY_UPDATE_INTERVAL_SECONDS = 1.5


@dataclass(frozen=True)
class ActivitySnapshot:
    """Safe public state for one background activity."""

    activity_id: str
    title: str
    status: str
    detail: str = ""
    result: str = ""


@runtime_checkable
class ActivitySurfaceProvider(Protocol):
    """Optional provider extension for ephemeral activity surfaces."""

    async def open_activity_surface(
        self,
        conversation_id: ConversationId,
        snapshot: ActivitySnapshot,
    ) -> str | None: ...

    async def update_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None: ...

    async def close_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None: ...

    def restore_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None: ...


class ActivityProjection:
    """Coalesce nonessential updates without slowing agent execution."""

    def __init__(
        self,
        provider: ActivitySurfaceProvider,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None:
        self.provider = provider
        self.conversation_id = conversation_id
        self.surface_id = surface_id
        self.pending: ActivitySnapshot | None = None
        self.task: asyncio.Task[None] | None = None
        self.closed = False

    def publish(self, snapshot: ActivitySnapshot) -> None:
        if self.closed:
            return
        self.pending = snapshot
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        await asyncio.sleep(ACTIVITY_UPDATE_INTERVAL_SECONDS)
        while not self.closed and self.pending is not None:
            snapshot = self.pending
            self.pending = None
            try:
                await self.provider.update_activity_surface(
                    self.conversation_id,
                    self.surface_id,
                    snapshot,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(
                    "activity",
                    "surface_update_error",
                    exc,
                    {"activity_id": snapshot.activity_id},
                )
            if self.pending is not None:
                await asyncio.sleep(ACTIVITY_UPDATE_INTERVAL_SECONDS)

    async def close(self, snapshot: ActivitySnapshot) -> bool:
        if self.closed:
            return True
        self.closed = True
        self.pending = None
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        try:
            await self.provider.close_activity_surface(
                self.conversation_id,
                self.surface_id,
                snapshot,
            )
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                "activity",
                "surface_close_error",
                exc,
                {"activity_id": snapshot.activity_id},
            )
            return False


_projections: dict[tuple[str, str], ActivityProjection] = {}
_close_tasks: set[asyncio.Task[None]] = set()


def _finish_close_task(task: asyncio.Task[None]) -> None:
    _close_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        log_exception("activity", "surface_close_task_error", exc)


def _projection_key(activity_id: str) -> tuple[str, str]:
    return session.current_runtime().key, activity_id


def _snapshot(
    record: dict[str, Any],
    *,
    detail: str = "",
) -> ActivitySnapshot:
    status = str(record.get("status") or "unknown")
    result = str(record.get("result") or record.get("error") or "")
    return ActivitySnapshot(
        activity_id=str(record.get("id") or ""),
        title=str(record.get("last_task") or record.get("task") or "Subagent"),
        status=status,
        detail=detail,
        result=result if status != "running" else "",
    )


async def open_subagent_activity(record: dict[str, Any]) -> None:
    """Open a best-effort activity surface for a background subagent."""
    provider = get_chat_provider()
    if not isinstance(provider, ActivitySurfaceProvider):
        return
    snapshot = _snapshot(record, detail="Starting")
    try:
        surface_id = await provider.open_activity_surface(
            record["chat_id"],
            snapshot,
        )
    except Exception as exc:
        log_exception(
            "activity",
            "surface_open_error",
            exc,
            {"activity_id": snapshot.activity_id},
        )
        return
    if not surface_id:
        return
    record["activity_surface_id"] = surface_id
    record["activity_surface_closed"] = False
    session.save_state()
    _projections[_projection_key(snapshot.activity_id)] = ActivityProjection(
        provider,
        record["chat_id"],
        surface_id,
    )


def publish_subagent_activity(record: dict[str, Any], detail: str) -> None:
    projection = _projections.get(_projection_key(str(record.get("id") or "")))
    if projection is not None:
        projection.publish(_snapshot(record, detail=detail))


async def close_subagent_activity(record: dict[str, Any]) -> None:
    activity_id = str(record.get("id") or "")
    surface_id = str(record.get("activity_surface_id") or "")
    if not surface_id or record.get("activity_surface_closed"):
        return
    provider = get_chat_provider()
    if not isinstance(provider, ActivitySurfaceProvider):
        return
    projection = _projections.pop(_projection_key(activity_id), None)
    snapshot = _snapshot(record)
    if projection is not None:
        closed = await projection.close(snapshot)
    else:
        try:
            await provider.close_activity_surface(
                record["chat_id"],
                surface_id,
                snapshot,
            )
            closed = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                "activity",
                "surface_recovery_close_error",
                exc,
                {"activity_id": activity_id},
            )
            closed = False
    if closed:
        record["activity_surface_closed"] = True
        session.save_state()


def schedule_subagent_activity_close(record: dict[str, Any]) -> None:
    """Close presentation asynchronously without gating agent capacity."""
    task = asyncio.create_task(close_subagent_activity(record))
    _close_tasks.add(task)
    task.add_done_callback(_finish_close_task)


async def reconcile_activity_surfaces() -> None:
    """Close surfaces left open by an interrupted process."""
    for runtime in session.all_chat_runtimes():
        with session.bound_runtime(runtime):
            for record in runtime.subagent_records.values():
                surface_id = str(record.get("activity_surface_id") or "")
                provider = get_chat_provider()
                if surface_id and isinstance(provider, ActivitySurfaceProvider):
                    provider.restore_activity_surface(
                        record["chat_id"],
                        surface_id,
                    )
                if (
                    surface_id
                    and not record.get("activity_surface_closed")
                    and record.get("status") != "running"
                ):
                    await close_subagent_activity(record)


async def close_activity_projections() -> None:
    """Stop transient update tasks during application shutdown."""
    close_tasks = tuple(_close_tasks)
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True)
    projections = tuple(_projections.values())
    _projections.clear()
    for projection in projections:
        projection.closed = True
        if projection.task is not None and not projection.task.done():
            projection.task.cancel()
    tasks = [
        projection.task
        for projection in projections
        if projection.task is not None
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
