"""Transient projection of root-agent tool activity."""

import asyncio
import json
import os
from typing import Any

from chat_provider import ConversationId
from config import CHAT_STREAM_INTERVAL
from goals import suspend_goal_pin, sync_goal_pin
from observability import log_exception
from presentation import show_pinned_status


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
    """Reuse the pinned status slot for one root turn's tool calls."""

    def __init__(self, conversation_id: ConversationId) -> None:
        self.conversation_id = conversation_id
        self.owner = "tool:" + os.urandom(6).hex()
        self.active = False
        self._pending: tuple[str, dict[str, Any]] | None = None
        self._projection_task: asyncio.Task[None] | None = None

    async def _project(
        self,
        status: str,
        call: dict[str, Any],
    ) -> None:
        try:
            await show_pinned_status(
                self.conversation_id,
                self.owner,
                status + "\n\n" + tool_preview(call),
            )
            if not self.active:
                self.active = True
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
