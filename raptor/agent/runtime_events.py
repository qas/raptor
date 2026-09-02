"""Typed events delivered from background runtime resources to the root."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from raptor.chat.chat_provider import ConversationId
from config import MAX_TOOL_OUTPUT

_TRUNCATION_MARKER = "\n... [runtime event truncated] ...\n"


class RuntimeEventKind(StrEnum):
    SHELL_COMPLETED = "shell_completed"
    SUBAGENT_COMPLETED = "subagent_completed"


@dataclass(frozen=True)
class RuntimeEvent:
    conversation_id: ConversationId
    kind: RuntimeEventKind
    content: str
    done: asyncio.Future[bool]
    is_active: Callable[[], bool] | None = None

    def prompt(self) -> str:
        prefix = f'<runtime_event type="{self.kind.value}">\n'
        suffix = "\n</runtime_event>"
        content = self.content
        available = MAX_TOOL_OUTPUT - len(prefix) - len(suffix)
        if len(content) > available:
            retained = max(0, available - len(_TRUNCATION_MARKER))
            head = retained // 2
            tail = retained - head
            content = (
                content[:head]
                + _TRUNCATION_MARKER
                + (content[-tail:] if tail else "")
            )
        return prefix + content + suffix
