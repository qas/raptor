"""Concurrent chat-provider multiplexer."""
import asyncio
from dataclasses import replace
from typing import Any

from chat_provider import (
    ChatEvent,
    ChatProvider,
    ConversationId,
    Controls,
    IncomingAction,
    IncomingMessage,
    MessageId,
    PollResult,
    ProviderCapabilities,
)
from observability import log_exception
from activity import (
    ActivityConversationProvider,
    ActivityFinishResult,
    ActivitySnapshot,
    ActivitySurfaceProvider,
)


class MultiProvider:
    """Route provider-qualified conversations to their native adapters."""

    authorized_user_id = "multi:operator"

    def __init__(
        self,
        providers: tuple[ChatProvider, ...],
    ) -> None:
        if not providers:
            raise ValueError("MultiProvider requires at least one provider")
        self.providers = providers
        self.by_name = {provider.name: provider for provider in providers}
        if len(self.by_name) != len(providers):
            raise ValueError("MultiProvider provider names must be unique")
        self.name = ",".join(provider.name for provider in providers)
        self.capabilities = self._combined_capabilities()
        primary = providers[0]
        raw_primary = primary.primary_conversation_id
        self.primary_conversation_id = (
            self._conversation_id(primary, raw_primary)
            if str(raw_primary).strip()
            else raw_primary
        )
        self._poll_tasks: dict[str, asyncio.Task[PollResult]] = {}
        self._cursors: dict[str, object | None] = {
            provider.name: None for provider in providers
        }
        self._event_routes: dict[int, tuple[ChatProvider, ChatEvent]] = {}
        self._cursor = 0

    def _combined_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            drafts=any(
                provider.capabilities.drafts for provider in self.providers
            ),
            reasoning_summaries=any(
                provider.capabilities.reasoning_summaries
                for provider in self.providers
            ),
            pins=any(
                provider.capabilities.pins for provider in self.providers
            ),
            controls=any(
                provider.capabilities.controls for provider in self.providers
            ),
            typing=any(
                provider.capabilities.typing for provider in self.providers
            ),
        )

    @staticmethod
    def _conversation_id(
        provider: ChatProvider,
        raw_id: ConversationId,
    ) -> str:
        encoded = provider.encode_conversation_id(raw_id)
        return f"{provider.name}:{encoded}"

    def _route_conversation(
        self,
        conversation_id: ConversationId,
    ) -> tuple[ChatProvider, ConversationId]:
        value = str(conversation_id)
        name, separator, raw_id = value.partition(":")
        provider = self.by_name.get(name)
        if not separator or provider is None:
            raise ValueError(f"unknown multiplexed conversation: {value}")
        return provider, provider.decode_conversation_id(raw_id)

    @staticmethod
    def encode_conversation_id(conversation_id: ConversationId) -> str:
        return str(conversation_id)

    @staticmethod
    def decode_conversation_id(value: str) -> ConversationId:
        return value

    def prepare_event(self, event: ChatEvent) -> None:
        route = self._event_routes.get(id(event))
        if route is None:
            raise ValueError("event was not produced by this provider")
        provider, raw_event = route
        provider.prepare_event(raw_event)

    def _route_action(self, action_id: str) -> tuple[ChatProvider, str]:
        name, separator, raw_id = str(action_id).partition(":")
        provider = self.by_name.get(name)
        if not separator or provider is None:
            raise ValueError(f"unknown multiplexed action: {action_id}")
        return provider, raw_id

    def capture_delivery_context(
        self,
        conversation_id: ConversationId,
    ) -> tuple[str, Any | None]:
        provider, raw_id = self._route_conversation(conversation_id)
        return (provider.name, provider.capture_delivery_context(raw_id))

    def activate_delivery_context(
        self,
        conversation_id: ConversationId,
        value: Any | None,
    ) -> tuple[ChatProvider, Any | None]:
        provider, raw_id = self._route_conversation(conversation_id)
        if value is None:
            return (
                provider,
                provider.activate_delivery_context(raw_id, None),
            )
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("invalid multiplexed delivery context")
        provider_name, nested = value
        if provider_name != provider.name:
            raise ValueError("delivery context and conversation providers differ")
        return (
            provider,
            provider.activate_delivery_context(raw_id, nested),
        )

    def restore_delivery_context(self, token: Any | None) -> None:
        if not isinstance(token, tuple) or len(token) != 2:
            raise ValueError("invalid multiplexed delivery context token")
        provider, nested = token
        provider.restore_delivery_context(nested)

    async def initialize(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        initialized: list[ChatProvider] = []
        try:
            for provider in self.providers:
                await provider.initialize(commands)
                initialized.append(provider)
            primary = self.providers[0]
            self.primary_conversation_id = self._conversation_id(
                primary,
                primary.primary_conversation_id,
            )
            self.capabilities = self._combined_capabilities()
        except Exception:
            for provider in reversed(initialized):
                try:
                    await provider.close()
                except Exception as close_exc:
                    log_exception(
                        provider.name,
                        "startup_cleanup_error",
                        close_exc,
                    )
            raise

    async def close(self) -> None:
        tasks = tuple(self._poll_tasks.values())
        self._poll_tasks.clear()
        self._event_routes.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        results = await asyncio.gather(
            *(provider.close() for provider in self.providers),
            return_exceptions=True,
        )
        for provider, result in zip(self.providers, results):
            if isinstance(result, BaseException):
                log_exception(
                    provider.name,
                    "shutdown_error",
                    result,
                )

    def _start_polls(self, timeout: int) -> None:
        for provider in self.providers:
            task = self._poll_tasks.get(provider.name)
            if task is None:
                self._poll_tasks[provider.name] = asyncio.create_task(
                    provider.poll(
                        self._cursors[provider.name],
                        timeout=timeout,
                    )
                )

    def _normalize(
        self,
        provider: ChatProvider,
        event: ChatEvent,
    ) -> ChatEvent:
        conversation_id = (
            self._conversation_id(provider, event.conversation_id)
            if event.conversation_id is not None
            else None
        )
        sender_id = (
            self.authorized_user_id
            if event.sender_id == provider.authorized_user_id
            else f"{provider.name}:{event.sender_id}"
        )
        if isinstance(event, IncomingMessage):
            assert conversation_id is not None
            normalized = replace(
                event,
                conversation_id=conversation_id,
                sender_id=sender_id,
            )
        elif isinstance(event, IncomingAction):
            presentation_conversation_id = (
                self._conversation_id(
                    provider,
                    event.presentation_conversation_id,
                )
                if event.presentation_conversation_id is not None
                else None
            )
            normalized = replace(
                event,
                action_id=f"{provider.name}:{event.action_id}",
                conversation_id=conversation_id,
                sender_id=sender_id,
                presentation_conversation_id=presentation_conversation_id,
            )
        else:
            normalized = event
        self._event_routes[id(normalized)] = (provider, event)
        return normalized

    async def poll(
        self,
        cursor: object | None,
        *,
        timeout: int,
    ) -> PollResult:
        del cursor
        while True:
            self._start_polls(timeout)
            done, _pending = await asyncio.wait(
                tuple(self._poll_tasks.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            selected = next(iter(done))
            provider_name = next(
                name
                for name, task in self._poll_tasks.items()
                if task is selected
            )
            self._poll_tasks.pop(provider_name, None)
            provider = self.by_name[provider_name]
            batch = selected.result()
            self._cursors[provider_name] = batch.cursor
            if not batch.events:
                continue
            events = tuple(
                self._normalize(provider, event)
                for event in batch.events
            )
            self._cursor += 1
            return PollResult(events, self._cursor)

    async def send_text(
        self,
        conversation_id: ConversationId,
        text: str,
    ) -> tuple[MessageId, ...]:
        provider, raw_id = self._route_conversation(conversation_id)
        message_ids = await provider.send_text(raw_id, text)
        return tuple(message_ids or ())

    async def send_draft(
        self,
        conversation_id: ConversationId,
        draft_id: int,
        text: str,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        if provider.capabilities.drafts:
            await provider.send_draft(raw_id, draft_id, text)

    async def send_reasoning_summary(
        self,
        conversation_id: ConversationId,
        delta: str,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        if provider.capabilities.reasoning_summaries:
            await provider.send_reasoning_summary(raw_id, delta)

    async def create_message(
        self,
        conversation_id: ConversationId,
        text: str,
        controls: Controls = (),
    ) -> MessageId:
        provider, raw_id = self._route_conversation(conversation_id)
        return await provider.create_message(raw_id, text, controls)

    async def edit_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
        text: str,
        controls: Controls = (),
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        await provider.edit_message(raw_id, message_id, text, controls)

    async def delete_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        await provider.delete_message(raw_id, message_id)

    async def delete_messages(
        self,
        conversation_id: ConversationId,
        message_ids: tuple[MessageId, ...],
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        await provider.delete_messages(raw_id, message_ids)

    async def pin_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        if provider.capabilities.pins:
            await provider.pin_message(raw_id, message_id)

    async def unpin_message(
        self,
        conversation_id: ConversationId,
        message_id: MessageId,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        if provider.capabilities.pins:
            await provider.unpin_message(raw_id, message_id)

    async def set_typing(
        self,
        conversation_id: ConversationId,
        active: bool,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        if provider.capabilities.typing:
            await provider.set_typing(raw_id, active)

    async def reject_busy_message(
        self,
        conversation_id: ConversationId,
    ) -> bool:
        provider, raw_id = self._route_conversation(conversation_id)
        return await provider.reject_busy_message(raw_id)

    async def acknowledge_queued_message(
        self,
        conversation_id: ConversationId,
    ) -> None:
        provider, raw_id = self._route_conversation(conversation_id)
        await provider.acknowledge_queued_message(raw_id)

    async def finish_event(self, event: ChatEvent) -> None:
        route = self._event_routes.pop(id(event), None)
        if route is None:
            raise ValueError("event was not produced by this provider")
        provider, raw_event = route
        await provider.finish_event(raw_event)

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None:
        provider, raw_id = self._route_action(action_id)
        await provider.answer_action(raw_id, text, alert=alert)

    async def open_activity_surface(
        self,
        conversation_id: ConversationId,
        snapshot: ActivitySnapshot,
        existing_surface_id: str | None = None,
    ) -> str | None:
        provider, raw_id = self._route_conversation(conversation_id)
        if not isinstance(provider, ActivitySurfaceProvider):
            return None
        nested_id: str | None = None
        if existing_surface_id is not None:
            nested_id = self._route_activity_surface(
                conversation_id,
                existing_surface_id,
            )[2]
        opened_id = await provider.open_activity_surface(
            raw_id,
            snapshot,
            nested_id,
        )
        if not opened_id:
            return None
        return f"{provider.name}:{opened_id}"

    def _route_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> tuple[ActivitySurfaceProvider, ConversationId, str]:
        provider, raw_id = self._route_conversation(conversation_id)
        name, separator, nested_id = surface_id.partition(":")
        if (
            not separator
            or name != provider.name
            or not isinstance(provider, ActivitySurfaceProvider)
        ):
            raise ValueError("activity surface and conversation differ")
        return provider, raw_id, nested_id

    def activity_surface_conversation_id(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> ConversationId:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        if not isinstance(provider, ActivityConversationProvider):
            return conversation_id
        nested_conversation_id = provider.activity_surface_conversation_id(
            raw_id,
            nested_id,
        )
        return self._conversation_id(provider, nested_conversation_id)

    async def update_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> None:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        await provider.update_activity_surface(raw_id, nested_id, snapshot)

    async def append_activity_message(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        text: str,
    ) -> None:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        await provider.append_activity_message(raw_id, nested_id, text)

    async def finish_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
        snapshot: ActivitySnapshot,
    ) -> ActivityFinishResult:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        return await provider.finish_activity_surface(
            raw_id,
            nested_id,
            snapshot,
        )

    async def delete_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        await provider.delete_activity_surface(raw_id, nested_id)

    def restore_activity_surface(
        self,
        conversation_id: ConversationId,
        surface_id: str,
    ) -> None:
        provider, raw_id, nested_id = self._route_activity_surface(
            conversation_id,
            surface_id,
        )
        provider.restore_activity_surface(raw_id, nested_id)
