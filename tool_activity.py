"""Transient projection of root-agent tool activity."""

import asyncio
import json
from collections import deque
from typing import Any

from chat_provider import ConversationId, Controls, MessageId
from chat_runtime import get_chat_provider
from config import CHAT_STREAM_INTERVAL
from observability import log_exception


MAX_RETAINED_TOOL_BUBBLES = 64


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
    """Project each tool call through one fresh canonical status bubble."""

    def __init__(
        self,
        conversation_id: ConversationId,
    ) -> None:
        self.conversation_id = conversation_id
        self.message_id: MessageId | None = None
        self._completed_message_ids: deque[MessageId] = deque()
        self._pending: tuple[str, dict[str, Any], Controls] | None = None
        self._projection_task: asyncio.Task[None] | None = None

    async def _project(
        self,
        status: str,
        call: dict[str, Any],
        controls: Controls = (),
    ) -> None:
        try:
            provider = get_chat_provider()
            effective_controls = (
                controls if provider.capabilities.controls else ()
            )
            text = status + "\n\n" + tool_preview(call)
            if self.message_id is not None:
                await provider.edit_message(
                    self.conversation_id,
                    self.message_id,
                    text,
                    effective_controls,
                )
            else:
                self.message_id = await provider.create_message(
                    self.conversation_id,
                    text,
                    effective_controls,
                )
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
            status, call, controls = self._pending
            self._pending = None
            await self._project(status, call, controls)

    def _queue(
        self,
        status: str,
        call: dict[str, Any],
        controls: Controls = (),
    ) -> None:
        self._pending = (status, dict(call), controls)
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

    async def approval(
        self,
        call: dict[str, Any],
        controls: Controls,
    ) -> int | str:
        self._queue("⚠️ Approval required", call, controls)
        await self._flush()
        if self.message_id is None:
            raise RuntimeError("Could not present tool approval")
        return self.message_id

    async def _delete_message(self, message_id: MessageId) -> None:
        try:
            await get_chat_provider().delete_message(
                self.conversation_id,
                message_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception("tool_activity", "bubble_cleanup_error", exc)

    async def finished(
        self,
        call: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        status = "Completed" if result.get("ok") else "Failed"
        if result.get("status") == "interrupted":
            status = "Interrupted"
        if result.get("approval") == "denied":
            status = "Denied"
        elif result.get("approval") == "superseded":
            status = "Interrupted"
        self._queue(status, call)
        await self._flush()
        message_id = self.message_id
        if message_id is not None:
            if len(self._completed_message_ids) >= MAX_RETAINED_TOOL_BUBBLES:
                oldest = self._completed_message_ids[0]
                await self._delete_message(oldest)
                self._completed_message_ids.popleft()
            self._completed_message_ids.append(message_id)
            self.message_id = None

    async def clear(self) -> None:
        task = self._projection_task
        self._pending = None
        self._projection_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.message_id is not None:
            await self._delete_message(self.message_id)
            self.message_id = None
        while self._completed_message_ids:
            message_id = self._completed_message_ids[-1]
            await self._delete_message(message_id)
            self._completed_message_ids.pop()
