"""Provider-neutral transient and persistent chat presentation policy."""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from chat_provider import (
    ActionButton,
    ConversationId,
    Controls,
    MessageId,
)
from chat_runtime import get_chat_provider
from observability import log_exception
import session

def _pinned_status_lock() -> asyncio.Lock:
    runtime = session.current_runtime()
    loop = asyncio.get_running_loop()
    if (
        runtime.presentation_lock is None
        or runtime.presentation_loop is not loop
    ):
        runtime.presentation_lock = asyncio.Lock()
        runtime.presentation_loop = loop
    return runtime.presentation_lock


async def _unpin_and_delete(
    conversation_id: ConversationId,
    message_id: MessageId,
) -> None:
    provider = get_chat_provider()
    if provider.capabilities.pins:
        try:
            await provider.unpin_message(conversation_id, message_id)
        except Exception as exc:
            log_exception("presentation", "unpin_error", exc)
    try:
        await provider.delete_message(conversation_id, message_id)
    except Exception as exc:
        log_exception("presentation", "delete_error", exc)


async def show_pinned_status(
    conversation_id: ConversationId,
    owner: str,
    text: str,
    *,
    controls: Controls = (),
) -> MessageId:
    """Project one owner into the shared persistent status surface."""
    async with _pinned_status_lock():
        provider = get_chat_provider()
        runtime = session.current_runtime()
        effective_controls = (
            controls if provider.capabilities.controls else ()
        )
        message_id = runtime.pinned_status_message_id
        same_conversation = (
            runtime.pinned_status_conversation_id == conversation_id
        )
        if message_id is not None and not same_conversation:
            old_conversation = runtime.pinned_status_conversation_id
            if old_conversation is not None:
                await _unpin_and_delete(old_conversation, message_id)
            runtime.pinned_status_conversation_id = None
            runtime.pinned_status_message_id = None
            runtime.pinned_status_owner = None
            message_id = None
        if message_id is not None and same_conversation:
            try:
                await provider.edit_message(
                    conversation_id,
                    message_id,
                    text,
                    effective_controls,
                )
                runtime.pinned_status_owner = owner
                return message_id
            except Exception as exc:
                log_exception("presentation", "status_edit_error", exc)
                await _unpin_and_delete(conversation_id, message_id)
                runtime.pinned_status_conversation_id = None
                runtime.pinned_status_message_id = None
                runtime.pinned_status_owner = None

        message_id = await provider.create_message(
            conversation_id,
            text,
            effective_controls,
        )
        if provider.capabilities.pins:
            try:
                await provider.pin_message(conversation_id, message_id)
            except Exception:
                try:
                    await provider.delete_message(
                        conversation_id,
                        message_id,
                    )
                except Exception as cleanup_exc:
                    log_exception(
                        "presentation",
                        "failed_pin_cleanup_error",
                        cleanup_exc,
                    )
                raise
        runtime.pinned_status_conversation_id = conversation_id
        runtime.pinned_status_message_id = message_id
        runtime.pinned_status_owner = owner
        return message_id


async def clear_pinned_status(
    conversation_id: ConversationId,
    *,
    owner: str | None = None,
) -> bool:
    async with _pinned_status_lock():
        runtime = session.current_runtime()
        if runtime.pinned_status_conversation_id != conversation_id:
            return False
        if owner is not None and runtime.pinned_status_owner != owner:
            return False
        message_id = runtime.pinned_status_message_id
        runtime.pinned_status_conversation_id = None
        runtime.pinned_status_message_id = None
        runtime.pinned_status_owner = None
        if message_id is not None:
            await _unpin_and_delete(conversation_id, message_id)
        return True


async def steering_indicator(
    conversation_id: ConversationId,
    steer_id: str,
) -> MessageId | None:
    provider = get_chat_provider()
    controls: Controls = ((
        ActionButton("⚡ Apply now", f"steer:{steer_id}:apply"),
        ActionButton("✖ Cancel", f"steer:{steer_id}:cancel"),
    ),)
    if not provider.capabilities.controls:
        controls = ()
    try:
        return await provider.create_message(
            conversation_id,
            "Steering queued for the next safe point.",
            controls,
        )
    except Exception as exc:
        log_exception("presentation", "steering_indicator_error", exc)
        return None


async def clear_steering_indicator(
    conversation_id: ConversationId,
    message_id: MessageId | None,
    steer_id: str | None = None,
) -> None:
    del steer_id
    if message_id is None:
        return
    try:
        await get_chat_provider().delete_message(
            conversation_id,
            message_id,
        )
    except Exception as exc:
        log_exception("presentation", "steering_cleanup_error", exc)


async def show_goal_pin(
    conversation_id: ConversationId,
    text: str,
    goal_id: str = "",
) -> MessageId:
    return await show_pinned_status(
        conversation_id,
        "goal:" + goal_id,
        text,
    )


async def clear_goal_pin(
    conversation_id: ConversationId,
    message_id: MessageId,
    goal_id: str = "",
) -> None:
    del message_id
    await clear_pinned_status(
        conversation_id,
        owner="goal:" + goal_id,
    )


async def _animate_compacting(
    conversation_id: ConversationId,
    message_id: MessageId,
    *,
    interval: float,
) -> None:
    provider = get_chat_provider()
    frames = ("Compacting..", "Compacting...", "Compacting.")
    frame = 0
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await provider.edit_message(
                    conversation_id,
                    message_id,
                    frames[frame],
                )
            except Exception as exc:
                log_exception("presentation", "compacting_animation_error", exc)
                return
            frame = (frame + 1) % len(frames)
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def compacting_indicator(
    conversation_id: ConversationId,
    *,
    interval: float = 1.2,
) -> AsyncIterator[None]:
    provider = get_chat_provider()
    message_id: MessageId | None = None
    animation: asyncio.Task[None] | None = None
    try:
        try:
            message_id = await provider.create_message(
                conversation_id,
                "Compacting.",
            )
            animation = asyncio.create_task(
                _animate_compacting(
                    conversation_id,
                    message_id,
                    interval=interval,
                )
            )
        except Exception as exc:
            log_exception("presentation", "compacting_indicator_error", exc)
        yield
    finally:
        if animation is not None:
            animation.cancel()
            try:
                await animation
            except asyncio.CancelledError:
                pass
        if message_id is not None:
            try:
                await provider.delete_message(
                    conversation_id,
                    message_id,
                )
            except Exception as exc:
                log_exception("presentation", "compacting_cleanup_error", exc)


async def typing_loop(conversation_id: ConversationId) -> None:
    provider = get_chat_provider()
    if not provider.capabilities.typing:
        return
    try:
        while True:
            try:
                await provider.set_typing(conversation_id, True)
            except Exception as exc:
                log_exception("presentation", "typing_error", exc)
                return
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        try:
            await provider.set_typing(conversation_id, False)
        except Exception as exc:
            log_exception("presentation", "typing_cleanup_error", exc)
