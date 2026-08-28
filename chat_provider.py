"""Provider-neutral contracts for interactive chat frontends."""
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, runtime_checkable

ConversationId: TypeAlias = int | str
MessageId: TypeAlias = int | str
UserId: TypeAlias = int | str


@dataclass(frozen=True)
class ActionButton:
    label: str
    action: str


Controls: TypeAlias = tuple[tuple[ActionButton, ...], ...]


@dataclass(frozen=True)
class ProviderCapabilities:
    drafts: bool = False
    reasoning_summaries: bool = False
    pins: bool = False
    controls: bool = False
    typing: bool = False


@dataclass(frozen=True)
class IncomingMessage:
    conversation_id: ConversationId
    sender_id: UserId
    message_id: MessageId
    text: str
    interactive: bool = True


@dataclass(frozen=True)
class IncomingAction:
    action_id: str
    conversation_id: ConversationId | None
    sender_id: UserId
    message_id: MessageId | None
    data: str
    interactive: bool = True


ChatEvent: TypeAlias = IncomingMessage | IncomingAction


@dataclass(frozen=True)
class PollResult:
    events: tuple[ChatEvent, ...]
    cursor: object | None


@runtime_checkable
class ChatProvider(Protocol):
    """Capabilities required by the agent framework from a chat backend."""

    name: str
    authorized_user_id: UserId
    primary_conversation_id: ConversationId
    capabilities: ProviderCapabilities

    def encode_conversation_id(self, conversation_id: ConversationId) -> str:
        """Serialize a native conversation ID for provider multiplexing."""
        ...

    def decode_conversation_id(self, value: str) -> ConversationId:
        """Restore a native conversation ID from its serialized form."""
        ...

    async def initialize(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None: ...

    async def close(self) -> None: ...

    async def poll(
        self,
        cursor: object | None,
        *,
        timeout: int,
    ) -> PollResult: ...

    async def send_text(
        self,
        conversation_id: ConversationId,
        text: str,
    ) -> tuple[MessageId, ...]:
        """Send text and return every persistent message ID created."""
        ...

    async def send_draft(
        self,
        conversation_id: ConversationId,
        draft_id: int,
        text: str,
    ) -> None: ...

    async def send_reasoning_summary(
        self,
        conversation_id: ConversationId,
        delta: str,
    ) -> None: ...

    async def create_message(
        self,
        conversation_id: ConversationId,
        text: str,
        controls: Controls = (),
    ) -> MessageId: ...

    async def edit_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
        text: str,
        controls: Controls = (),
    ) -> None: ...

    async def delete_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None: ...

    async def pin_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None: ...

    async def unpin_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None: ...

    async def set_typing(
        self,
        conversation_id: ConversationId,
        active: bool,
    ) -> None: ...

    async def reject_busy_message(
        self,
        conversation_id: ConversationId,
    ) -> bool:
        """Terminate a request-style input instead of queueing chat steering."""
        ...

    async def acknowledge_queued_message(
        self,
        conversation_id: ConversationId,
    ) -> None:
        """Finish transport bookkeeping after an input becomes steering."""
        ...

    async def finish_event(self, event: ChatEvent) -> None:
        """Finalize transport bookkeeping after core event handling ends."""
        ...

    def prepare_event(self, event: ChatEvent) -> None:
        """Bind transport state immediately before core event handling."""
        ...

    def capture_delivery_context(
        self,
        conversation_id: ConversationId,
    ) -> Any | None:
        """Capture provider state required to answer this input later."""
        ...

    def activate_delivery_context(
        self,
        conversation_id: ConversationId,
        delivery_context: Any | None,
    ) -> Any | None:
        """Activate captured state, or detach it with ``None``."""
        ...

    def restore_delivery_context(self, token: Any | None) -> None:
        """Restore provider state after deferred delivery."""
        ...

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None: ...
