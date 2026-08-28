"""Process-wide chat-provider binding used by provider-neutral core code."""
import importlib
from contextlib import contextmanager
from typing import Any, Iterator

from chat_provider import ChatProvider, ConversationId

_provider: ChatProvider | None = None


def set_chat_provider(
    provider: ChatProvider | None,
) -> ChatProvider | None:
    """Bind the process-wide provider and return the previous binding."""
    global _provider
    previous = _provider
    _provider = provider
    return previous


def load_chat_provider(spec: str) -> ChatProvider:
    """Load a built-in provider or a ``module:attribute`` plugin."""
    if spec == "telegram":
        from telegram import telegram_provider
        return telegram_provider
    if spec == "responses_api":
        from responses_provider import responses_provider
        return responses_provider
    if ":" not in spec:
        raise ValueError(
            "provider must be 'telegram', 'responses_api', or "
            "'module:attribute'"
        )
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    provider = candidate() if callable(candidate) else candidate
    if not isinstance(provider, ChatProvider):
        raise TypeError(
            f"{spec} does not implement the ChatProvider contract"
        )
    return provider


def load_chat_providers(specs: tuple[str, ...]) -> ChatProvider:
    """Load one provider directly or compose several concurrently."""
    providers = tuple(load_chat_provider(spec) for spec in specs)
    if not providers:
        raise ValueError("CHAT_PROVIDERS must contain at least one provider")
    if len(providers) == 1:
        return providers[0]
    from multi_provider import MultiProvider
    return MultiProvider(providers)


def get_chat_provider() -> ChatProvider:
    if _provider is None:
        raise RuntimeError("chat provider has not been initialized")
    return _provider


def capture_delivery_context(
    conversation_id: ConversationId,
) -> Any | None:
    return get_chat_provider().capture_delivery_context(conversation_id)


def activate_delivery_context(
    conversation_id: ConversationId,
    delivery_context: Any | None,
) -> Any | None:
    if delivery_context is None:
        return None
    return get_chat_provider().activate_delivery_context(
        conversation_id,
        delivery_context,
    )


def restore_delivery_context(token: Any | None) -> None:
    if token is None:
        return
    get_chat_provider().restore_delivery_context(token)


@contextmanager
def bound_delivery_context(
    conversation_id: ConversationId,
    delivery_context: Any | None,
) -> Iterator[None]:
    token = activate_delivery_context(conversation_id, delivery_context)
    try:
        yield
    finally:
        restore_delivery_context(token)


@contextmanager
def detached_delivery_context(
    conversation_id: ConversationId,
) -> Iterator[None]:
    """Temporarily send outside any request-correlated response slot."""
    provider = get_chat_provider()
    token = provider.activate_delivery_context(conversation_id, None)
    try:
        yield
    finally:
        provider.restore_delivery_context(token)


async def send(
    conversation_id: ConversationId,
    text: str,
) -> tuple[int | str, ...]:
    provider = get_chat_provider()
    message_ids = await provider.send_text(conversation_id, text)
    if message_ids is None:
        # Compatibility with lightweight third-party providers while the
        # tracked-delivery contract rolls out.
        return ()
    tracked = tuple(message_ids)
    if tracked:
        try:
            import session
            from chat_store import (
                append_meta,
                latest_active_user_turn_seq,
                session_exists,
            )

            session_id = str(session.state.get("current_session_id") or "")
            if session_id and session_exists(session_id):
                user_turn_seq = latest_active_user_turn_seq(session_id)
                if user_turn_seq is None:
                    return tracked
                append_meta(
                    session_id,
                    "chat_delivery",
                    {
                        "conversation_id": provider.encode_conversation_id(
                            conversation_id
                        ),
                        "message_ids": list(tracked),
                        "user_turn_seq": user_turn_seq,
                    },
                )
        except Exception:
            # Delivery already succeeded. A bookkeeping failure must not make
            # the caller retry and duplicate the visible message.
            pass
    return tracked


async def send_draft(
    conversation_id: ConversationId,
    draft_id: int,
    text: str,
) -> None:
    provider = get_chat_provider()
    if not provider.capabilities.drafts:
        return
    await provider.send_draft(conversation_id, draft_id, text)


async def send_reasoning_summary(
    conversation_id: ConversationId,
    delta: str,
) -> None:
    provider = get_chat_provider()
    if not provider.capabilities.reasoning_summaries:
        return
    await provider.send_reasoning_summary(conversation_id, delta)
