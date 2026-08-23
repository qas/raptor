"""Raptor application lifecycle."""

import asyncio
import os
import signal

import httpx

import session
from chat_runtime import load_chat_providers, send, set_chat_provider
from chat_store import chat_path, ensure_chat_dirs
from config import (
    CHAT_DIR,
    CHAT_PROVIDERS,
    CHAT_STREAMING,
    CONTEXT_COMPACT_RATIO,
    CONTEXT_SAFETY_TOKENS,
    MAX_TOOL_ROUNDS,
    MODEL_CONTEXT_TOKENS,
    RAPTOR_HOME,
    STATE_PATH,
    SUBAGENT_MODEL_CONTEXT_TOKENS,
    context_input_budget,
    subagent_context_input_budget,
)
from controller import ensure_root_session
from goals import goal_is_active, prepare_goal_on_startup
from loop import COMMANDS, handle_event
from responses import ensure_model
from runtime import clear_runtime_if_ours, set_runtime
from session import bootstrap_runtime_storage, rehydrate_pending_inputs, state
from shell_sessions import cancel_shell_sessions, shell_completion_event_loop
from skills import close_skill_discovery, start_skill_discovery
from subagents import (
    cancel_background_subagents,
    completion_event_loop,
)
from thread_state import thread_active
from thread_status import ensure_thread_status
from observability import log_event


async def main() -> None:
    loop = asyncio.get_running_loop()
    current_task = asyncio.current_task()

    if current_task is not None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, current_task.cancel)
            except NotImplementedError:
                pass

    provider = load_chat_providers(CHAT_PROVIDERS)
    session.responses = httpx.AsyncClient(timeout=None)
    set_chat_provider(provider)
    subagent_event_task: asyncio.Task[None] | None = None
    shell_event_task: asyncio.Task[None] | None = None
    cursor: object | None = None

    try:
        # Metadata discovery overlaps provider/model startup. Full skill
        # bodies remain unloaded until a turn invokes read_skill.
        start_skill_discovery()
        await provider.initialize(COMMANDS)
        await ensure_model()

        ensure_chat_dirs()
        RAPTOR_HOME.mkdir(parents=True, exist_ok=True)
        CHAT_DIR.mkdir(parents=True, exist_ok=True)
        storage = bootstrap_runtime_storage()

        subagent_event_task = asyncio.create_task(completion_event_loop())
        shell_event_task = asyncio.create_task(shell_completion_event_loop())
        set_runtime(daemon=session.DAEMON_MODE)

        session_id = state.get("current_session_id")
        conversation_id = provider.primary_conversation_id
        rehydrated = rehydrate_pending_inputs(conversation_id)
        log_event(
            "runtime",
            "ready",
            {
                "model": state["model"],
                "chat_provider": provider.name,
                "user": provider.authorized_user_id,
                "conversation": conversation_id,
                "session_id": session_id,
                "chat_path": (
                    str(chat_path(str(session_id)))
                    if session_id
                    else None
                ),
                "context_limit": MODEL_CONTEXT_TOKENS,
                "context_compact_ratio": CONTEXT_COMPACT_RATIO,
                "context_input_budget": context_input_budget(),
                "subagent_context_limit": SUBAGENT_MODEL_CONTEXT_TOKENS,
                "subagent_context_input_budget": (
                    subagent_context_input_budget()
                ),
                "context_safety_tokens": CONTEXT_SAFETY_TOKENS,
                "tool_round_limit": MAX_TOOL_ROUNDS,
                "streaming": CHAT_STREAMING,
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
                "rehydrated_steers": rehydrated,
            },
        )

        goal_notice = prepare_goal_on_startup()
        if goal_notice:
            try:
                await send(conversation_id, goal_notice)
            except Exception as exc:
                log_event(
                    "goal",
                    "startup_notice_error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        if thread_active():
            try:
                await ensure_thread_status(conversation_id)
            except Exception as exc:
                log_event(
                    "thread",
                    "thread_pin_error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        if (goal_is_active() and not thread_active()) or rehydrated:
            ensure_root_session(conversation_id, None)

        while True:
            try:
                batch = await provider.poll(cursor, timeout=50)
                cursor = batch.cursor
                for event in batch.events:
                    try:
                        provider.prepare_event(event)
                        await handle_event(event)
                    finally:
                        await provider.finish_event(event)

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
        if session.active_task and not session.active_task.done():
            session.active_task.cancel()

        if subagent_event_task is not None:
            subagent_event_task.cancel()
        if shell_event_task is not None:
            shell_event_task.cancel()

        await cancel_background_subagents()
        await cancel_shell_sessions()

        event_tasks = tuple(
            task
            for task in (subagent_event_task, shell_event_task)
            if task is not None
        )
        if event_tasks:
            await asyncio.gather(*event_tasks, return_exceptions=True)

        await close_skill_discovery()
        clear_runtime_if_ours()
        await provider.close()
        set_chat_provider(None)
        await session.responses.aclose()
