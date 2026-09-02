"""Raptor application lifecycle."""

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from raptor.state import session
from raptor.chat.activity import (
    close_activity_projections,
    reconcile_activity_surfaces,
)
from raptor.app.application_control import (
    ExitRequest,
    activate_application_exit,
    bind_application_task,
    current_exit_request,
    discard_exit_request,
    unbind_application_task,
)
from raptor.agent.agent import flush_pending_delivery, repair_interrupted_root_turn
from raptor.chat.chat_provider import ChatEvent, ChatProvider
from raptor.chat.chat_runtime import load_chat_providers, send, set_chat_provider
from raptor.state.chat_store import chat_path, ensure_chat_dirs
from config import (
    CHAT_DIR,
    CHAT_PROVIDERS,
    CHAT_STREAMING,
    CHAT_TOOL_ACTIVITY,
    CONTEXT_COMPACT_RATIO,
    CONTEXT_SAFETY_TOKENS,
    MAX_TOOL_ROUNDS,
    RAPTOR_HOME,
    STATE_PATH,
    model_context_input_budget,
)
from raptor.agent.controller import ensure_root_session, interrupt_root_turn
from raptor.shell.console_follow import close_follow_console
from raptor.agent.goals import goal_is_active, pause_goal, prepare_goal_on_startup
from raptor.chat.loop import COMMANDS, accepts_event, handle_event
from network import outbound_http_client
from observability import log_event, log_exception
from raptor.model.model_providers import MODEL_CONFIGURATION
from raptor.model.responses import ensure_target
from raptor.state.session import bootstrap_runtime_storage, rehydrate_pending_inputs, state
from raptor.shell.shell_sessions import cancel_shell_sessions
from raptor.agent.skills import (
    close_skill_discovery,
    initialize_builtin_skills,
    start_skill_discovery,
)
from raptor.agent.subagents import (
    cancel_background_subagents,
    restore_pending_subagent_completions,
)
from raptor.agent.thread_state import thread_active
from raptor.agent.thread_status import ensure_thread_status
from raptor.app.workspace_identity import initialize_workspace_identity


async def dispatch_event(provider: ChatProvider, event: ChatEvent) -> None:
    """Finalize every transport event and bind state only after admission."""
    try:
        provider.prepare_event(event)
        conversation_id = event.conversation_id
        if conversation_id is None or not accepts_event(event, provider):
            return
        with session.bound_chat(conversation_id):
            await handle_event(event)
    finally:
        try:
            await provider.finish_event(event)
        except BaseException:
            discard_exit_request()
            raise
        if activate_application_exit():
            # Deliver cancellation before the dispatcher can admit another
            # event from the same provider batch.
            await asyncio.sleep(0)


async def dispatch_events(
    provider: ChatProvider,
    events: tuple[ChatEvent, ...],
) -> None:
    """Dispatch every accepted transport event independently."""
    for event in events:
        try:
            await dispatch_event(provider, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                provider.name,
                "event_error",
                exc,
                {"conversation_id": str(event.conversation_id)},
            )


async def _cleanup(name: str, operation: Awaitable[Any]) -> None:
    """Run one shutdown operation without skipping later resource owners."""
    try:
        await operation
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log_exception("runtime", "shutdown_error", exc, {"operation": name})


def _should_pause_active_goal() -> bool:
    return bool(
        current_exit_request() is not ExitRequest.RESTART
        and goal_is_active()
        and not thread_active()
    )


async def main(on_ready: Callable[[], None] | None = None) -> None:
    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()

    if current_task is not None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, current_task.cancel)
            except NotImplementedError:
                pass

    provider = load_chat_providers(CHAT_PROVIDERS)
    session.responses = outbound_http_client(
        timeout=httpx.Timeout(None, connect=10.0),
    )
    set_chat_provider(provider)
    cursor: object | None = None

    try:
        if current_task is not None:
            bind_application_task(current_task)
        initialize_workspace_identity()
        initialize_builtin_skills()
        default_target = await ensure_target(MODEL_CONFIGURATION.default_target)
        session.set_default_model_target(default_target)
        # Metadata discovery overlaps provider/model startup. Full skill
        # bodies remain unloaded until a turn invokes read_skill.
        start_skill_discovery()
        await provider.initialize(COMMANDS)
        ensure_chat_dirs()
        RAPTOR_HOME.mkdir(parents=True, exist_ok=True)
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        primary_runtime = session.set_default_chat(
            provider.primary_conversation_id
        )
        storage = bootstrap_runtime_storage()
        conversation_id = provider.primary_conversation_id
        rehydrated_by_chat: dict[str, int] = {}
        for runtime in session.all_chat_runtimes():
            with session.bound_runtime(runtime):
                rehydrated_by_chat[runtime.key] = rehydrate_pending_inputs(
                    runtime.conversation_id
                )
        with session.bound_runtime(primary_runtime):
            session_id = state.get("current_session_id")
            primary_target = session.current_model_target()
            rehydrated = rehydrated_by_chat.get(primary_runtime.key, 0)
        for runtime in session.all_chat_runtimes():
            with session.bound_runtime(runtime):
                root_interrupted = repair_interrupted_root_turn()
                delivery_ready = await flush_pending_delivery(
                    runtime.conversation_id
                )
                goal_notice = prepare_goal_on_startup(
                    root_interrupted=root_interrupted,
                )
                if delivery_ready and goal_notice:
                    try:
                        await send(runtime.conversation_id, goal_notice)
                    except Exception as exc:
                        log_event(
                            "goal",
                            "startup_notice_error",
                            {
                                "chat": runtime.key,
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
                if delivery_ready and thread_active():
                    try:
                        await ensure_thread_status(runtime.conversation_id)
                    except Exception as exc:
                        log_event(
                            "thread",
                            "thread_pin_error",
                            {
                                "chat": runtime.key,
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
        await reconcile_activity_surfaces()
        for runtime in session.all_chat_runtimes():
            with session.bound_runtime(runtime):
                pending_completions = restore_pending_subagent_completions()
                restored = rehydrated_by_chat.get(runtime.key, 0)
                if (
                    goal_is_active() and not thread_active()
                ) or restored or pending_completions:
                    ensure_root_session(runtime.conversation_id, None)

        log_event(
            "runtime",
            "ready",
            {
                "model_provider": primary_target.provider_id,
                "model": primary_target.model,
                "chat_provider": provider.name,
                "user": provider.authorized_user_id,
                "conversation": conversation_id,
                "session_id": session_id,
                "chat_path": (
                    str(chat_path(str(session_id))) if session_id else None
                ),
                "context_limit": (
                    MODEL_CONFIGURATION.provider(primary_target.provider_id)
                    .settings_for(primary_target.model)
                    .context_window
                ),
                "context_compact_ratio": CONTEXT_COMPACT_RATIO,
                "context_input_budget": model_context_input_budget(
                    MODEL_CONFIGURATION.provider(primary_target.provider_id)
                    .settings_for(primary_target.model)
                    .context_window
                ),
                "context_safety_tokens": CONTEXT_SAFETY_TOKENS,
                "tool_round_limit": MAX_TOOL_ROUNDS,
                "streaming": CHAT_STREAMING,
                "tool_activity": CHAT_TOOL_ACTIVITY,
                "capabilities": {
                    "drafts": provider.capabilities.drafts,
                    "reasoning_summaries": (
                        provider.capabilities.reasoning_summaries
                    ),
                    "pins": provider.capabilities.pins,
                    "controls": provider.capabilities.controls,
                    "typing": provider.capabilities.typing,
                },
                "pid": os.getpid(),
                "daemon": session.DAEMON_MODE,
                "state": str(STATE_PATH),
                "home": str(RAPTOR_HOME),
                "repaired_chats": storage["repaired_chats"],
                "created_sessions": storage["created_sessions"],
                "rehydrated_steers": rehydrated,
            },
        )
        if on_ready is not None:
            on_ready()

        while True:
            try:
                batch = await provider.poll(cursor, timeout=50)
                cursor = batch.cursor
                await dispatch_events(provider, batch.events)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    provider.name,
                    "poll_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                await asyncio.sleep(2)

    finally:
        await _cleanup("close followed console", close_follow_console())
        for runtime in session.all_chat_runtimes():
            with session.bound_runtime(runtime):
                try:
                    if _should_pause_active_goal():
                        pause_goal()
                except Exception as exc:
                    log_exception(
                        "runtime",
                        "shutdown_error",
                        exc,
                        {"operation": "pause goal"},
                    )
                await _cleanup("interrupt root turn", interrupt_root_turn())

        for runtime in session.all_chat_runtimes():
            with session.bound_runtime(runtime):
                await _cleanup(
                    "cancel background subagents",
                    cancel_background_subagents(),
                )
                await _cleanup("cancel shell sessions", cancel_shell_sessions())
        await _cleanup("close activity projections", close_activity_projections())

        await _cleanup("close skill discovery", close_skill_discovery())
        await _cleanup("close chat provider", provider.close())
        set_chat_provider(None)
        await _cleanup("close Responses client", session.responses.aclose())
        if current_task is not None:
            unbind_application_task(current_task)
