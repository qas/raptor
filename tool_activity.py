"""Transient projection of root-agent tool activity."""

import asyncio
import json
import os
from typing import Any

from chat_provider import ConversationId
from chat_runtime import get_chat_provider
from config import CHAT_STREAM_INTERVAL
from goals import suspend_goal_pin, sync_goal_pin
from observability import log_exception
from presentation import (
    create_pinned_status,
    delete_pinned_status,
    show_pinned_status,
)


def tool_preview(call: dict[str, Any]) -> str:
    """Render the bounded call preview shared with tool approval."""
    name = str(call.get("name") or "unknown")
    raw_arguments = call.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
        rendered = json.dumps(
            arguments,
            indent=2,
            ensure_ascii=False,
        )
    except (TypeError, json.JSONDecodeError):
        rendered = str(raw_arguments)
    if len(rendered) > 3200:
        rendered = rendered[:3150] + "\n... [truncated]"
    return "Tool: " + name + "\n\n" + rendered


class ToolActivitySurface:
    """Project one agent turn's tools through the canonical status bubble."""

    def __init__(
        self,
        conversation_id: ConversationId,
        *,
        isolated: bool = False,
    ) -> None:
        self.conversation_id = conversation_id
        self.isolated = isolated
        self.owner = "tool:" + os.urandom(6).hex()
        self.active = False
        self.message_id: int | str | None = None
        self._pending: tuple[str, dict[str, Any]] | None = None
        self._projection_task: asyncio.Task[None] | None = None

    async def _project(
        self,
        status: str,
        call: dict[str, Any],
    ) -> None:
        try:
            text = status + "\n\n" + tool_preview(call)
            if self.isolated and self.message_id is not None:
                await get_chat_provider().edit_message(
                    self.conversation_id,
                    self.message_id,
                    text,
                )
            elif self.isolated:
                self.message_id = await create_pinned_status(
                    self.conversation_id,
                    text,
                )
            else:
                await show_pinned_status(
                    self.conversation_id,
                    self.owner,
                    text,
                )
            if not self.active:
                self.active = True
                if not self.isolated:
                    await suspend_goal_pin(self.conversation_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception("tool_activity", "projection_error", exc)

    async def _drain(self) -> None:
        first = True
        while self._pending is not None:
            if not first:
                await asyncio.sleep(CHAT_STREAM_INTERVAL)
            first = False
            status, call = self._pending
            self._pending = None
            await self._project(status, call)

    def _queue(self, status: str, call: dict[str, Any]) -> None:
        self._pending = (status, dict(call))
        if self._projection_task is None or self._projection_task.done():
            self._projection_task = asyncio.create_task(self._drain())

    async def _flush(self) -> None:
        task = self._projection_task
        if task is not None:
            await task

    async def stream(
        self,
        call: dict[str, Any],
        complete: bool,
    ) -> None:
        self._queue("Preparing tool", call)
        if complete:
            await self._flush()

    async def running(self, call: dict[str, Any]) -> None:
        self._queue("Running", call)
        await self._flush()

    async def finished(
        self,
        call: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        status = "Completed" if result.get("ok") else "Failed"
        if result.get("status") == "interrupted":
            status = "Interrupted"
        self._queue(status, call)
        await self._flush()

    async def clear(self) -> None:
        task = self._projection_task
        self._pending = None
        self._projection_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if not self.active:
            return
        try:
            if self.isolated:
                if self.message_id is not None:
                    await delete_pinned_status(
                        self.conversation_id,
                        self.message_id,
                    )
                    self.message_id = None
            else:
                await sync_goal_pin(
                    self.conversation_id,
                    released_owner=self.owner,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception("tool_activity", "cleanup_error", exc)
            return
        self.active = False
