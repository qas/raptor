"""Provider-neutral projection of subagent activity."""

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from raptor.state import session
from raptor.chat.chat_provider import ConversationId
from raptor.chat.chat_runtime import get_chat_provider
from config import MAX_TOOL_OUTPUT
from observability import log_exception

ACTIVITY_UPDATE_INTERVAL_SECONDS = 1.5
MAX_ACTIVITY_FIELD_CHARS = 600
MAX_ACTIVITY_MESSAGE_CHARS = MAX_TOOL_OUTPUT
MAX_ACTIVITY_STREAM_CHARS = MAX_TOOL_OUTPUT


def _bounded_activity_field(value: Any) -> str:
    text = str(value or "")
    if len(text) <= MAX_ACTIVITY_FIELD_CHARS:
        return text
    return text[: MAX_ACTIVITY_FIELD_CHARS - 3] + "..."


def _bounded_activity_stream(value: Any) -> str:
    text = str(value or "")
    if len(text) <= MAX_ACTIVITY_STREAM_CHARS:
        return text
    return "..." + text[-(MAX_ACTIVITY_STREAM_CHARS - 3):]


def _bounded_activity_message(value: Any) -> str:
    return str(value or "")[:MAX_ACTIVITY_MESSAGE_CHARS]


@dataclass(frozen=True)
class ActivitySnapshot:
    """Safe public state for one subagent activity surface."""

    activity_id: str
    title: str
    status: str
    generation: int = 1
    detail: str = ""
    result: str = ""
    reasoning_summary: str = ""
    reply: str = ""


@dataclass(frozen=True)
class ActivityFinishResult:
    """Durable progress from finalizing one activity run."""

    finished: bool
    result_delivered: bool


@runtime_checkable
class ActivitySurfaceProvider(Protocol):
    """Optional provider extension for isolated subagent output."""

    async def open_activity_surface(
        self,
        conversation_id: ConversationId,
        snapshot: ActivitySnapshot,
        existing_surface_id: str | None = None,
    ) -> str | None: ...

    async def update_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None: ...

    async def append_activity_message(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        text: str,
    ) -> None: ...

    async def finish_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> ActivityFinishResult: ...

    async def delete_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None: ...

    def restore_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None: ...


@runtime_checkable
class ActivityConversationProvider(Protocol):
    """Resolve an activity surface's interactive conversation."""

    def activity_surface_conversation_id(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> ConversationId: ...


class ActivityProjection:
    """Coalesce nonessential updates without slowing agent execution."""

    def __init__(
        self,
        provider: ActivitySurfaceProvider,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None:
        self.provider = provider
        self.conversation_id = conversation_id
        self.surface_id = surface_id
        self.pending: ActivitySnapshot | None = None
        self.last_published = snapshot
        self.desired = snapshot
        self.task: asyncio.Task[None] | None = None
        self.closed = False

    def publish(self, snapshot: ActivitySnapshot) -> None:
        if self.closed:
            return
        self.desired = snapshot
        if snapshot == self.pending:
            return
        if self.pending is None and snapshot == self.last_published:
            return
        self.pending = snapshot
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._drain())

    def publish_response(
        self,
        *,
        reasoning_summary: str | None = None,
        reply: str | None = None,
    ) -> None:
        current = self.desired
        self.publish(
            replace(
                current,
                reasoning_summary=(
                    current.reasoning_summary
                    if reasoning_summary is None
                    else _bounded_activity_stream(reasoning_summary)
                ),
                reply=(
                    current.reply
                    if reply is None
                    else _bounded_activity_stream(reply)
                ),
            )
        )

    def publish_activity(self, snapshot: ActivitySnapshot) -> None:
        current = self.desired
        self.publish(
            replace(
                snapshot,
                reasoning_summary=current.reasoning_summary,
                reply=current.reply,
            )
        )

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
                self.last_published = snapshot
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

    async def finish(self, snapshot: ActivitySnapshot) -> ActivityFinishResult:
        if not self.closed:
            self.closed = True
            self.pending = None
            if self.task is not None and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        try:
            return await self.provider.finish_activity_surface(
                self.conversation_id,
                self.surface_id,
                snapshot,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                "activity",
                "surface_finish_error",
                exc,
                {"activity_id": snapshot.activity_id},
            )
            return ActivityFinishResult(False, False)


_projections: dict[tuple[str, str, int], ActivityProjection] = {}


def _projection_key(activity_id: str, generation: int) -> tuple[str, str, int]:
    return session.current_runtime().key, activity_id, generation


def _snapshot(
    record: dict[str, Any],
    *,
    detail: str = "",
) -> ActivitySnapshot:
    status = str(record.get("status") or "unknown")
    result = str(record.get("result") or record.get("error") or "")
    return ActivitySnapshot(
        activity_id=_bounded_activity_field(record.get("id")),
        title=_bounded_activity_message(
            record.get("last_task") or record.get("task") or "Subagent"
        ),
        status=_bounded_activity_field(status),
        generation=max(1, int(record.get("run_generation") or 1)),
        detail=_bounded_activity_field(detail),
        result=(
            _bounded_activity_message(result)
            if status != "running"
            else ""
        ),
    )


async def open_subagent_activity(record: dict[str, Any]) -> None:
    """Open a best-effort activity surface for a subagent."""
    try:
        provider = get_chat_provider()
    except Exception as exc:
        log_exception("activity", "surface_provider_error", exc)
        return
    if not isinstance(provider, ActivitySurfaceProvider):
        return
    snapshot = _snapshot(record, detail="Starting")
    existing_surface_id = str(record.get("activity_surface_id") or "") or None
    try:
        surface_id = await provider.open_activity_surface(
            record["chat_id"],
            snapshot,
            existing_surface_id,
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
    activity_id = snapshot.activity_id
    record["activity_surface_id"] = surface_id
    try:
        session.save_state()
    except Exception:
        record["activity_surface_id"] = existing_surface_id
        if surface_id != existing_surface_id:
            try:
                await provider.delete_activity_surface(
                    record["chat_id"],
                    surface_id,
                )
            except Exception as exc:
                log_exception(
                    "activity",
                    "surface_open_rollback_error",
                    exc,
                    {"activity_id": activity_id},
                )
        raise
    _projections[
        _projection_key(activity_id, snapshot.generation)
    ] = ActivityProjection(provider, record["chat_id"], surface_id, snapshot)


def publish_subagent_activity(record: dict[str, Any], detail: str) -> None:
    snapshot = _snapshot(record, detail=detail)
    projection = _projections.get(
        _projection_key(snapshot.activity_id, snapshot.generation)
    )
    if projection is not None:
        projection.publish_activity(snapshot)


async def append_subagent_activity_input(
    record: dict[str, Any],
    text: str,
) -> None:
    """Append one parent-authored input to an existing activity surface."""
    surface_id = str(record.get("activity_surface_id") or "")
    provider = get_chat_provider()
    if not surface_id or not isinstance(provider, ActivitySurfaceProvider):
        return
    try:
        await provider.append_activity_message(
            record["chat_id"],
            surface_id,
            _bounded_activity_message(text),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception(
            "activity",
            "surface_input_error",
            exc,
            {"activity_id": _bounded_activity_field(record.get("id"))},
        )


def subagent_activity_conversation_id(
    record: dict[str, Any],
    *,
    fallback: ConversationId | None = None,
) -> ConversationId:
    """Return the provider-owned child conversation when one exists."""
    conversation_id = record.get("chat_id") or fallback
    if conversation_id is None:
        raise ValueError("subagent activity requires a parent conversation")
    surface_id = str(record.get("activity_surface_id") or "")
    if not surface_id:
        return conversation_id
    try:
        provider = get_chat_provider()
        if not isinstance(provider, ActivityConversationProvider):
            return conversation_id
        return provider.activity_surface_conversation_id(
            conversation_id,
            surface_id,
        )
    except Exception as exc:
        log_exception(
            "activity",
            "surface_conversation_error",
            exc,
            {"activity_id": _bounded_activity_field(record.get("id"))},
        )
        return conversation_id


def publish_subagent_response(
    record: dict[str, Any],
    *,
    reasoning_summary: str | None = None,
    reply: str | None = None,
) -> None:
    """Project safe model-visible output without exposing child context."""
    snapshot = _snapshot(record)
    projection = _projections.get(
        _projection_key(snapshot.activity_id, snapshot.generation)
    )
    if projection is not None:
        projection.publish_response(
            reasoning_summary=reasoning_summary,
            reply=reply,
        )


async def finish_subagent_activity(
    record: dict[str, Any],
    *,
    expected_generation: int | None = None,
) -> bool:
    """Finalize one run while preserving its continuable surface."""
    activity_id = _bounded_activity_field(record.get("id"))
    generation = max(1, int(record.get("run_generation") or 1))
    if expected_generation is not None and generation != expected_generation:
        return False
    if int(record.get("activity_finished_generation") or 0) >= generation:
        return True
    surface_id = str(record.get("activity_surface_id") or "")
    if not surface_id:
        return True
    provider = get_chat_provider()
    if not isinstance(provider, ActivitySurfaceProvider):
        return True
    projection = _projections.pop(
        _projection_key(activity_id, generation),
        None,
    )
    snapshot = _snapshot(record)
    if record.get("activity_result_delivered"):
        snapshot = replace(snapshot, result="")
    if projection is not None:
        current = projection.desired
        snapshot = replace(
            snapshot,
            reasoning_summary=current.reasoning_summary,
            reply=current.reply,
        )
        result = await projection.finish(snapshot)
    else:
        try:
            result = await provider.finish_activity_surface(
                record["chat_id"],
                surface_id,
                snapshot,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                "activity",
                "surface_finish_error",
                exc,
                {"activity_id": activity_id},
            )
            return False
    if result.result_delivered:
        record["activity_result_delivered"] = True
    if result.finished:
        record["activity_finished_generation"] = generation
    if result.result_delivered or result.finished:
        session.save_state()
    return result.finished


async def delete_subagent_activity(record: dict[str, Any]) -> bool:
    """Delete a terminal subagent's provider-owned activity surface."""
    surface_id = str(record.get("activity_surface_id") or "")
    if not surface_id:
        return True
    provider = get_chat_provider()
    if not isinstance(provider, ActivitySurfaceProvider):
        return False
    try:
        await provider.delete_activity_surface(
            record["chat_id"],
            surface_id,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception(
            "activity",
            "surface_delete_error",
            exc,
            {"activity_id": _bounded_activity_field(record.get("id"))},
        )
        return False
    record["activity_surface_id"] = None
    record.pop("activity_finished_generation", None)
    record.pop("activity_result_delivered", None)
    session.save_state()
    return True


async def reconcile_activity_surfaces() -> None:
    """Restore continuable surfaces after process restart."""
    for runtime in session.all_chat_runtimes():
        with session.bound_runtime(runtime):
            for record in runtime.subagent_records.values():
                surface_id = str(record.get("activity_surface_id") or "")
                provider = get_chat_provider()
                if (
                    surface_id
                    and isinstance(provider, ActivitySurfaceProvider)
                ):
                    try:
                        provider.restore_activity_surface(
                            record["chat_id"],
                            surface_id,
                        )
                    except Exception as exc:
                        log_exception(
                            "activity",
                            "surface_restore_error",
                            exc,
                            {
                                "activity_id": _bounded_activity_field(
                                    record.get("id")
                                )
                            },
                        )
                        continue
                    if record.get("status") != "running":
                        await finish_subagent_activity(record)


async def close_activity_projections() -> None:
    """Stop transient update tasks during application shutdown."""
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
