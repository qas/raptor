"""Provider-neutral text commands."""
import asyncio
import copy
import os
import re
from datetime import datetime, timezone
from typing import Any

from chat_provider import ConversationId

from chat_store import (
    TruncationCleanupError,
    append_meta,
    create_session,
    end_session,
    iter_events,
    list_sessions,
    materialize_session_truncation,
    new_session_id as generate_session_id,
    plan_session_truncation,
    session_contains_text,
    session_exists,
    session_summary,
)
from config import (
    AGENT_WORKDIR,
    CONTEXT_COMPACT_RATIO,
    CONTEXT_SAFETY_TOKENS,
    RAPTOR_PROXY,
    CHAT_STREAMING,
    MAX_TOOL_ROUNDS,
    model_context_input_budget,
)
from session import pending_approvals, save_state, state
import session
from observability import log_exception
from agent import context_tokens
from context import session_context_stats
from controller import (
    discard_runtime_events,
    ensure_root_session,
    interrupt_active_goal_controller,
    interrupt_root_turn,
    requeue_deferred_completions,
    session_transition_busy,
    start_manual_compaction,
    start_root_session,
)
from goals import (
    GOAL_ACTIVE,
    GOAL_BLOCKED,
    GOAL_PAUSED,
    clear_goal,
    complete_goal,
    current_goal,
    ensure_goal_pin,
    format_goal_status,
    goal_is_active,
    pause_goal,
    remove_goal_pin,
    replace_goal,
    resume_goal,
    todo_store_for_display,
)
from runtime import runtime_uptime
from chat_runtime import (
    bound_delivery_context,
    capture_delivery_context,
    get_chat_provider,
    send,
)
from engine import (
    function_call_output,
    interrupted_tool_result,
    response_calls,
    response_output,
    response_text,
)
from approval import approval_enabled, execute_tool_with_approval
from model_providers import MODEL_CONFIGURATION, ModelSettings, ModelTarget
from responses import (
    MODEL_LIST_TIMEOUT_SECONDS,
    list_models,
    model_provider,
    stateless_response,
)
from subagents import (
    cancel_background_subagents,
    pending_subagent_completions,
    running_background_subagents,
    subagent_summaries,
)
from shell_sessions import (
    cancel_shell_sessions,
    pending_shell_completions,
    running_shell_sessions,
)
from skills import skill_catalog_instructions
from steering import cancel_pending_steers
from threads import (
    finish_thread,
    resume_main_goal,
    start_thread,
)
from thread_state import current_thread, thread_active
from turn_runtime import TurnKind, turns
from todos import validate_plan
from version import display_version
from tool_activity import ToolActivitySurface

# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------


async def _run_stateless_ask(
    chat_id: ConversationId,
    prompt: str,
    delivery_context: object,
) -> None:
    current_task = asyncio.current_task()
    try:
        with bound_delivery_context(chat_id, delivery_context):
            tool_activity = (
                None if approval_enabled() else ToolActivitySurface(chat_id)
            )
            try:
                target = session.current_model_target()
                ask_instructions = await skill_catalog_instructions()
                work: list[dict] = [{"role": "user", "content": prompt}]
                tool_rounds = 0
                while True:
                    response = await stateless_response(
                        target,
                        work,
                        extra_instructions=ask_instructions,
                    )
                    output = response_output(response)
                    calls = response_calls(response)
                    if not calls:
                        answer = response_text(response)
                        if not answer:
                            raise RuntimeError("Model returned no text")
                        break
                    if MAX_TOOL_ROUNDS and tool_rounds >= MAX_TOOL_ROUNDS:
                        raise RuntimeError("Configured tool-round limit reached")
                    tool_rounds += 1
                    work.extend(output)
                    for call in calls:
                        execution_context = dict(state)
                        execution_context["session_id"] = state.get(
                            "current_session_id"
                        )
                        execution_context["model_target"] = target.to_dict()
                        execution_context["todo_state"] = state
                        if tool_activity is not None:
                            await tool_activity.running(call)
                        try:
                            result = await execute_tool_with_approval(
                                chat_id,
                                call,
                                execution_context=execution_context,
                            )
                        except asyncio.CancelledError:
                            if tool_activity is not None:
                                await tool_activity.finished(
                                    call,
                                    interrupted_tool_result(),
                                )
                            raise
                        except Exception:
                            if tool_activity is not None:
                                await tool_activity.finished(
                                    call,
                                    {"ok": False},
                                )
                            raise
                        if tool_activity is not None:
                            await tool_activity.finished(call, result)
                        work.append(function_call_output(call, result))
            except asyncio.CancelledError:
                await send(chat_id, "Ask cancelled.")
                raise
            except Exception as exc:
                await send(chat_id, f"Ask error: {type(exc).__name__}: {exc}")
            else:
                await send(chat_id, answer)
            finally:
                if tool_activity is not None:
                    await tool_activity.clear()
    finally:
        if turns.finish(current_task):
            ensure_root_session(chat_id, None)


def format_todos() -> str:
    todos = todo_store_for_display().get(
        "todos",
        [],
    )

    if not todos:
        return "No todos."

    marks = {
        "pending":
            "[ ]",
        "in_progress":
            "[>]",
        "completed":
            "[x]",
    }

    return "\n".join(
        (
            f"{marks.get(item['status'], '[ ]')} "
            f"{item['step']}"
        )
        for item in todos
    )


def format_subagents() -> str:
    rows = subagent_summaries()
    if not rows:
        return "No subagents."
    lines = ["Subagents:"]
    for row in rows:
        target = row.get("model_target")
        if isinstance(target, dict):
            provider = str(target.get("provider_id") or "unknown")
            model = str(target.get("model") or "unknown")
            model_label = f"{provider}/{model}"
        else:
            model_label = "unknown/unknown"
        mode = "background" if row.get("background") else "foreground"
        pending = " · result pending" if row.get("completion_pending") else ""
        task = " ".join(
            str(row.get("last_task") or row.get("task") or "").split()
        )
        if len(task) > 100:
            task = task[:99] + "…"
        task_suffix = f" — {task}" if task else ""
        lines.append(
            f"{row.get('id')} [{row.get('status')}] · {mode} · "
            f"{model_label}{pending}{task_suffix}"
        )
    return "\n".join(lines)


def _main_chat_sessions(
    query: str = "",
    *,
    current_id: str = "",
) -> list[dict[str, Any]]:
    needle = query.casefold()
    chat_key = session.current_runtime().key
    discovery_limit = 100 if needle else 20
    rows = sorted(
        (
            row
            for row in list_sessions(
                limit=discovery_limit,
                chat_key=chat_key,
                kinds={"main"},
            )
        ),
        key=lambda row: float(row.get("started_at") or 0),
        reverse=True,
    )
    if not needle and current_id:
        current = session_summary(current_id)
        if (
            current is not None
            and current.get("kind") == "main"
            and current.get("chat_key") == chat_key
        ):
            rows = [
                current,
                *(
                    row
                    for row in rows
                    if row.get("session_id") != current_id
                ),
            ][:20]
    matches: list[dict[str, Any]] = []
    for row in rows:
        if needle and not session_contains_text(
            str(row["session_id"]),
            needle,
        ):
            continue
        matches.append(row)
        if len(matches) >= 20:
            break
    return matches


def _format_chat_sessions(
    rows: list[dict[str, Any]],
    current_id: str,
) -> str:
    if not rows:
        return "No matching chats."
    lines = []
    for row in rows:
        started = datetime.fromtimestamp(
            float(row.get("started_at") or 0),
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M UTC")
        marker = " (current)" if row["session_id"] == current_id else ""
        lines.append(f"{row['session_id']} · {started}{marker}")
    return "Chats:\n" + "\n".join(lines)


def _archived_todos(session_id: str) -> list[dict[str, Any]]:
    archived: dict[str, Any] | None = None
    for event in iter_events(session_id):
        if event.get("type") == "session_end":
            archived = event
    if archived is not None:
        try:
            return validate_plan(archived.get("todos"))
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid archived plan in session {session_id}: {exc}"
            ) from exc
    return []


async def command(
    chat_id: ConversationId,
    text: str,
) -> bool | str:
    first, *rest = (
        text.strip().split(
            maxsplit=1
        )
    )

    cmd = first.split(
        "@",
        1,
    )[0].lower()

    arg = (
        rest[0].strip()
        if rest
        else ""
    )

    if cmd == "/help":
        await send(
            chat_id,
            (
                "/new - start a new session (archive previous)\n"
                "/chats [term] - list or search prior sessions\n"
                "/resume <session-id> - resume a prior session\n"
                "/status - show status\n"
                "/stop - interrupt the current root turn\n"
                "/stop all - also stop background agents and shells\n"
                "/compact - compact context\n"
                "/truncate <n> - remove the last n user turns\n"
                "/ask <message> - isolated tool-capable query\n"
                "/thread - fork into a temporary conversation\n"
                "/thread <message> - fork or continue, then send\n"
                "/thread status|clear|merge\n"
                "/models [provider] - list provider models\n"
                "/model - show current target and providers\n"
                "/model <provider> <model> - switch target\n"
                "/approval - show approval mode\n"
                "/approval on - require approval for side effects\n"
                "/approval off - allow tools immediately\n"
                "/todos - show todo list\n"
                "/subagents - show subagent status\n"
                "/goal - show persistent goal\n"
                "/goal <objective> - start or replace goal\n"
                "/goal pause|resume|complete|clear"
            ),
        )

        return True

    if cmd == "/ask":
        if not arg:
            await send(chat_id, "Usage: /ask <message>")
            return True
        if turns.is_running():
            await send(chat_id, "The agent is already running. Use /stop first.")
            return True
        delivery_context = capture_delivery_context(chat_id)
        turns.start(
            _run_stateless_ask(chat_id, arg, delivery_context),
            kind=TurnKind.STATELESS_ASK,
        )
        return True

    if cmd == "/thread":
        action = arg.lower()
        thread = current_thread()
        if action in {"", "status"} and thread:
            await send(
                chat_id,
                (
                    "Thread active.\n"
                    f"Branch: {str(thread.get('session_id') or '')[-8:]}\n"
                    "Use /thread clear or /thread merge."
                ),
            )
            return True
        if action == "status":
            await send(chat_id, "No thread is active.")
            return True
        if not action:
            result = await start_thread(chat_id)
            if result["ok"]:
                await send(
                    chat_id,
                    "Thread started. Normal messages now use the branch.",
                )
            else:
                await send(chat_id, str(result["error"]))
            return True
        if action not in {"clear", "merge"}:
            if not thread:
                result = await start_thread(chat_id)
                if not result["ok"]:
                    await send(chat_id, str(result["error"]))
                    return True
                await send(
                    chat_id,
                    "Thread started. Normal messages now use the branch.",
                )
            return arg
        result = await finish_thread(
            chat_id,
            merge=action == "merge",
        )
        if not result["ok"]:
            await send(chat_id, str(result["error"]))
            return True
        if action == "merge":
            await send(
                chat_id,
                f"Thread merged ({result['merged_items']} events).",
            )
        else:
            await send(chat_id, "Thread cleared.")
        resume_main_goal(chat_id)
        return True

    if cmd == "/stop":
        stop_all = arg.lower() == "all"
        if arg and not stop_all:
            await send(chat_id, "Usage: /stop [all]")
            return True
        paused = False
        if goal_is_active() and not thread_active():
            _goal, changed = pause_goal()
            paused = changed
            if paused:
                await remove_goal_pin(chat_id)
        interrupt = await interrupt_root_turn()
        if interrupt.error is not None:
            log_exception("agent", "stop_wait_error", interrupt.error)

        deferred_results = 0
        cancelled_subagents = 0
        cancelled_shells = 0
        cancelled_steers = 0
        if stop_all:
            cancelled_steers = await cancel_pending_steers()
            discard_runtime_events()
            deferred_results = (
                pending_subagent_completions()
                + pending_shell_completions()
            )
            cancelled_subagents = await cancel_background_subagents(
                discard_pending=True,
            )
            cancelled_shells = await cancel_shell_sessions()

        if (
            interrupt.interrupted
            or cancelled_subagents
            or cancelled_shells
            or cancelled_steers
            or deferred_results
            or paused
        ):
            message = (
                "Stop requested; cleanup is still finishing."
                if interrupt.interrupted and not interrupt.completed
                else "Stopped."
            )
            if paused:
                message += " Active goal paused."
            if cancelled_subagents:
                message += (
                    "\nCancelled background subagents: "
                    f"{cancelled_subagents}"
                )
            if cancelled_shells:
                message += (
                    "\nCancelled background shells: "
                    f"{cancelled_shells}"
                )
            if cancelled_steers:
                message += f"\nDiscarded queued steers: {cancelled_steers}"
            if deferred_results:
                message += (
                    "\nDiscarded pending background results: "
                    f"{deferred_results}"
                )
            await send(
                chat_id,
                message,
            )

        else:
            await send(
                chat_id,
                "Nothing is running.",
            )

        return True

    if cmd == "/status":
        target = session.current_model_target()
        try:
            target_provider = model_provider(target)
            target_settings = target_provider.settings_for(target.model)
            models = (
                await asyncio.wait_for(
                    list_models(target.provider_id, max_retries=0),
                    timeout=MODEL_LIST_TIMEOUT_SECONDS,
                )
            )

            responses_status = "up"

        except Exception as exc:
            log_exception("responses", "model_list_error", exc)
            target_settings = ModelSettings()
            models = []
            responses_status = (
                "unconfigured"
                if target.provider_id not in MODEL_CONFIGURATION.providers
                else "down/unreachable"
            )

        running = turns.is_running()
        task_age = turns.elapsed_seconds()

        context_window = target_settings.context_window
        budget = model_context_input_budget(context_window)
        session_id = state.get("current_session_id") or "(none)"
        stats = (
            session_context_stats(str(session_id))
            if session_id and session_id != "(none)"
            and session_exists(str(session_id))
            else {
                "archive_events": 0,
                "checkpoint": False,
                "checkpoint_through": None,
                "active_native_events": 0,
            }
        )
        if context_window:
            context_limit_line = (
                f"context estimate: ~{context_tokens():,} tokens\n"
                f"context limit: {context_window:,} tokens\n"
                f"compact threshold: {budget:,} tokens "
                f"({int(CONTEXT_COMPACT_RATIO * 100)}%)\n"
                f"safety reserve: {CONTEXT_SAFETY_TOKENS:,} tokens"
            )
        else:
            context_limit_line = (
                f"context estimate: ~{context_tokens():,} tokens\n"
                f"context limit: unknown\n"
                f"auto-compact: disabled"
            )
        checkpoint_line = (
            f"checkpoint: yes\n"
            f"checkpoint through: {stats['checkpoint_through']}"
            if stats["checkpoint"]
            else "checkpoint: no"
        )
        reasoning_effort = target_settings.reasoning_effort or "(model default)"
        interrupted_subagents = len(state.get("interrupted_subagents", []))
        await send(
            chat_id,
            (
                f"version: {display_version()}\n"
                f"chat provider: {get_chat_provider().name}\n"
                f"model provider: {target.provider_id}\n"
                f"Responses: {responses_status}\n"
                f"proxy: {'enabled' if RAPTOR_PROXY else 'disabled'}\n"
                f"model: {target.model}\n"
                f"reasoning effort: {reasoning_effort}\n"
                f"served models: {len(models)}\n"
                f"pid: {os.getpid()}\n"
                f"process: {'daemon' if session.DAEMON_MODE else 'foreground'}\n"
                f"uptime: {runtime_uptime()}s\n"
                f"running task: {'yes' if running else 'no'}\n"
                f"task age: {task_age}s\n"
                f"pending steers: {len(session.pending_steers)}\n"
                f"thread: {'active' if thread_active() else 'none'}\n"
                f"approval: {state.get('approval_mode', 'off')}\n"
                f"pending approvals: {len(pending_approvals)}\n"
                f"background subagents: {running_background_subagents()}\n"
                f"background shells: {running_shell_sessions()}\n"
                f"pending background results: "
                f"{pending_subagent_completions() + pending_shell_completions()}\n"
                f"subagent threads: {len(session.subagent_records)}\n"
                f"interrupted subagents: {interrupted_subagents}\n"
                f"{format_goal_status()}\n"
                f"session: {session_id}\n"
                f"archive events: {stats['archive_events']}\n"
                f"{checkpoint_line}\n"
                f"active native events: {stats['active_native_events']}\n"
                f"{context_limit_line}\n"
                f"todos: {len(state.get('todos', []))}\n"
                f"workdir: {AGENT_WORKDIR}\n"
                f"streaming: {'on' if CHAT_STREAMING else 'off'}"
            ),
        )

        return True

    if cmd == "/goal":
        if thread_active():
            await send(
                chat_id,
                "Persistent goals are unavailable inside a thread.",
            )
            return True
        action = arg.lower()
        if not arg:
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            status = str(goal.get("status") or "")
            objective = str(
                goal.get("objective") or ""
            )
            message = (
                f"Goal: {status}\n{objective}"
            )
            if (
                status == GOAL_BLOCKED
                and goal.get("blocked_reason")
            ):
                message += (
                    "\nBlocked: "
                    f"{goal['blocked_reason']}"
                )
            await send(chat_id, message)
            return True
        if action == "pause":
            goal, changed = pause_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            if not changed:
                await send(
                    chat_id,
                    (
                        "Goal is "
                        f"{goal.get('status')}; "
                        "nothing to pause."
                    ),
                )
                return True
            await remove_goal_pin(chat_id)
            await interrupt_active_goal_controller(
                str(goal.get("id") or "")
            )
            await send(
                chat_id,
                "Goal paused.",
            )
            return True
        if action == "resume":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            if goal.get("status") not in {
                GOAL_PAUSED,
                GOAL_BLOCKED,
            }:
                await send(
                    chat_id,
                    (
                        "Goal is "
                        f"{goal.get('status')}; "
                        "nothing to resume."
                    ),
                )
                return True
            resume_goal()
            await requeue_deferred_completions()
            await ensure_goal_pin(chat_id)
            if not turns.is_running():
                start_root_session(chat_id, None)
            await send(
                chat_id,
                "Goal resumed.",
            )
            return True
        if action == "complete":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            complete_goal(str(goal["id"]))
            await remove_goal_pin(chat_id)
            await send(
                chat_id,
                f"Goal complete: {goal.get('objective')}",
            )
            return True
        if action == "clear":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            goal_id = str(goal.get("id") or "")
            clear_goal()
            await remove_goal_pin(chat_id)
            await interrupt_active_goal_controller(
                goal_id
            )
            await send(chat_id, "Goal cleared.")
            return True
        existing = current_goal()
        if (
            existing
            and existing.get("status")
            in {
                GOAL_ACTIVE,
                GOAL_PAUSED,
                GOAL_BLOCKED,
            }
        ):
            await send(
                chat_id,
                (
                    "An unfinished goal already exists. "
                    "Clear or complete it first.\n"
                    f"Current ({existing.get('status')}): "
                    f"{existing.get('objective')}"
                ),
            )
            return True
        try:
            goal = replace_goal(arg)
        except ValueError as exc:
            await send(chat_id, str(exc))
            return True
        await ensure_goal_pin(chat_id)
        await send(
            chat_id,
            f"Goal started: {goal['objective']}",
        )
        if not turns.is_running():
            start_root_session(chat_id, None)
        else:
            ensure_root_session(chat_id, None)
        return True

    if cmd == "/chats":
        current_id = str(state.get("current_session_id") or "")
        await send(
            chat_id,
            _format_chat_sessions(
                _main_chat_sessions(arg, current_id=current_id),
                current_id,
            ),
        )
        return True

    if cmd == "/resume":
        if not arg:
            await send(chat_id, "Usage: /resume <session-id>")
            return True
        if thread_active():
            await send(chat_id, "Clear or merge the active thread first.")
            return True
        if session_transition_busy():
            await send(chat_id, "Busy. Use /stop all first.")
            return True
        target = session_summary(arg)
        if target is not None and (
            target.get("kind") != "main"
            or target.get("chat_key") != session.current_runtime().key
        ):
            target = None
        if target is None:
            await send(chat_id, f"Chat not found: {arg}")
            return True
        current_id = str(state.get("current_session_id") or "")
        if arg == current_id:
            await send(chat_id, f"Already using chat: {arg}")
            return True
        try:
            resumed_target = ModelTarget.from_value(target.get("model_target"))
            MODEL_CONFIGURATION.validate_target(resumed_target)
            # Preflight before archiving the currently usable session.
            model_provider(resumed_target).api_key()
        except (RuntimeError, ValueError) as exc:
            await send(chat_id, f"Cannot resume chat: {exc}")
            return True
        if current_id and session_exists(current_id):
            end_session(
                current_id,
                reason="session_switched",
                todos=list(state.get("todos") or []),
            )
        state["current_session_id"] = arg
        session.set_current_model_target(resumed_target)
        state["todos"] = _archived_todos(arg)
        state["pending_inputs"] = []
        state["active_root_turn"] = None
        state["interrupted_subagents"] = []
        append_meta(arg, "session_resumed", {"from_session_id": current_id})
        save_state()
        await ensure_goal_pin(chat_id)
        await send(chat_id, f"Resumed chat: {arg}")
        return True

    if cmd == "/new":
        if thread_active():
            await send(
                chat_id,
                "Clear or merge the active thread first.",
            )
            return True
        if session_transition_busy():
            await send(
                chat_id,
                "Busy. Use /stop all first.",
            )
            return True
        old_session_id = state.get("current_session_id")
        if old_session_id and session_exists(str(old_session_id)):
            end_session(
                str(old_session_id),
                reason="new_session",
                todos=list(state.get("todos") or []),
            )
        new_session_id = create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=session.current_model_target().to_dict(),
        )
        state["current_session_id"] = new_session_id
        state["todos"] = []
        state["pending_inputs"] = []
        state["active_root_turn"] = None
        state["interrupted_subagents"] = []
        save_state()
        short_id = str(new_session_id)[-8:]
        await send(
            chat_id,
            (
                f"New session: {short_id}\n"
                "Previous transcript archived."
            ),
        )
        return True

    if cmd == "/truncate":
        parts = arg.split()
        if len(parts) != 1:
            await send(chat_id, "Usage: /truncate <positive integer>")
            return True
        if re.fullmatch(r"[0-9]+", parts[0]) is None:
            await send(chat_id, "Usage: /truncate <positive integer>")
            return True
        try:
            turns_to_remove = int(parts[0])
        except ValueError:
            await send(chat_id, "Usage: /truncate <positive integer>")
            return True
        if turns_to_remove <= 0:
            await send(chat_id, "Usage: /truncate <positive integer>")
            return True
        if thread_active():
            await send(chat_id, "Clear or merge the active thread first.")
            return True
        if session_transition_busy():
            await send(chat_id, "Busy. Use /stop all first.")
            return True
        current_id = str(state.get("current_session_id") or "")
        pending_delivery = state.get("pending_delivery")
        if pending_delivery is not None:
            await send(
                chat_id,
                "A response is awaiting delivery; try /truncate again later.",
            )
            return True
        if not current_id or not session_exists(current_id):
            await send(chat_id, "Cannot truncate: no active session.")
            return True
        try:
            truncation_plan = plan_session_truncation(
                current_id,
                turns=turns_to_remove,
                chat_key=session.current_runtime().key,
                model_target=session.current_model_target().to_dict(),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            await send(chat_id, f"Could not truncate session: {exc}")
            return True
        for _attempt in range(8):
            new_id = generate_session_id()
            if not session_exists(new_id):
                break
        else:
            await send(chat_id, "Could not allocate a truncation session.")
            return True
        old_marker = state.get("session_transition")
        preparing_marker = {
            "kind": "history_truncate",
            "phase": "preparing",
            "source_session_id": current_id,
            "destination_session_id": str(new_id),
            "turns": turns_to_remove,
            "copied_items": 0,
        }
        state["session_transition"] = preparing_marker
        try:
            save_state()
        except (OSError, RuntimeError, ValueError) as exc:
            state["session_transition"] = old_marker
            await send(chat_id, f"Could not truncate session: {exc}")
            return True
        try:
            candidate_created = False
            _new_id, copied_items = materialize_session_truncation(
                truncation_plan,
                new_id,
            )
            candidate_created = True
            append_meta(
                str(new_id),
                "history_truncated",
                {"source_session_id": current_id, "turns": turns_to_remove},
            )
            marker = {
                **preparing_marker,
                "phase": "committed",
                "copied_items": copied_items,
            }
            state["current_session_id"] = str(new_id)
            state["session_transition"] = marker
            save_state()
        except TruncationCleanupError as exc:
            state["current_session_id"] = current_id
            state["session_transition"] = preparing_marker
            log_exception("commands", "truncate_cleanup_error", exc)
            await send(
                chat_id,
                "Could not truncate session; restart recovery is pending.",
            )
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            state["current_session_id"] = current_id
            state["session_transition"] = preparing_marker
            candidate = session_summary(str(new_id))
            owned_candidate = bool(
                candidate
                and candidate.get("kind") == "main"
                and candidate.get("chat_key") == session.current_runtime().key
                and candidate.get("parent_session_id") == current_id
            )
            cleanup_complete = True
            try:
                if (
                    (candidate_created or owned_candidate)
                    and session_exists(str(new_id))
                ):
                    end_session(str(new_id), reason="history_truncate_failed")
            except (OSError, RuntimeError, ValueError) as cleanup_exc:
                cleanup_complete = False
                log_exception("commands", "truncate_cleanup_error", cleanup_exc)
            if cleanup_complete:
                try:
                    state["session_transition"] = old_marker
                    save_state()
                except (OSError, RuntimeError, ValueError) as cleanup_exc:
                    state["session_transition"] = preparing_marker
                    log_exception(
                        "commands",
                        "truncate_marker_cleanup_error",
                        cleanup_exc,
                    )
            else:
                state["session_transition"] = preparing_marker
            await send(chat_id, f"Could not truncate session: {exc}")
            return True
        try:
            end_session(
                current_id,
                reason="history_truncated",
                todos=list(state.get("todos") or []),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            log_exception("commands", "truncate_archive_error", exc)
            await send(
                chat_id,
                (
                    f"Truncated the last {turns_to_remove} user turn(s), but "
                    "could not archive the old transcript. Files and tool "
                    "side effects were not reverted; the old transcript "
                    "remains available as audit history."
                ),
            )
            return True
        try:
            append_meta(
                str(new_id),
                "history_truncated_complete",
                {"source_session_id": current_id},
            )
            old_records = copy.deepcopy(state.get("subagents") or {})
            for record in (state.get("subagents") or {}).values():
                if (
                    isinstance(record, dict)
                    and str(record.get("parent_session_id") or "")
                    == current_id
                ):
                    record.setdefault("origin_parent_session_id", current_id)
                    record["parent_session_id"] = str(new_id)
            state["session_transition"] = None
            save_state()
        except (OSError, RuntimeError, ValueError) as exc:
            log_exception("commands", "truncate_completion_error", exc)
            state["session_transition"] = marker
            if "old_records" in locals():
                state["subagents"] = old_records
            await send(
                chat_id,
                "Truncation completed, but recovery metadata remains pending. "
                "Files and tool side effects were not reverted.",
            )
            return True
        deleted_messages = 0
        failed_deletions = 0
        if truncation_plan.chat_messages:
            provider = get_chat_provider()
            expected_conversation = provider.encode_conversation_id(chat_id)
            for encoded_conversation, message_id in truncation_plan.chat_messages:
                if encoded_conversation != expected_conversation:
                    failed_deletions += 1
                    log_exception(
                        "commands",
                        "truncate_message_owner_error",
                        ValueError("message reference crosses conversations"),
                    )
                    continue
                try:
                    await provider.delete_message(chat_id, message_id)
                except Exception as exc:
                    failed_deletions += 1
                    log_exception(
                        "commands",
                        "truncate_message_delete_error",
                        exc,
                        {"message_id": message_id},
                    )
                else:
                    deleted_messages += 1
        if failed_deletions:
            deletion_status = (
                f"Deleted {deleted_messages} linked chat message(s); "
                f"{failed_deletions} could not be deleted."
            )
        elif deleted_messages:
            deletion_status = (
                f"Deleted {deleted_messages} linked chat message(s)."
            )
        else:
            deletion_status = (
                "No linked chat messages were available to delete; older "
                "turns may predate message tracking."
            )
        await send(
            chat_id,
            (
                f"Truncated the last {turns_to_remove} user turn(s).\n"
                f"{deletion_status}\n"
                "Files and tool side effects were not reverted; the old "
                "transcript remains available as audit history."
            ),
        )
        return True

    if cmd == "/compact":
        if turns.is_running():
            await send(
                chat_id,
                "Busy. Use /stop first.",
            )
            return True
        start_manual_compaction(chat_id)
        return True

    if cmd == "/approval":
        if not arg:
            await send(
                chat_id,
                (
                    "Approval: "
                    f"{state.get('approval_mode', 'off')}\n"
                    "Protected tools: shell, write_file, edit_file"
                ),
            )
            return True
        mode = arg.lower()
        if mode not in {"on", "off"}:
            await send(
                chat_id,
                "Usage: /approval on | /approval off",
            )
            return True
        state["approval_mode"] = mode
        save_state()
        sid = state.get("current_session_id")
        if sid:
            append_meta(
                str(sid),
                "approval_changed",
                {"mode": mode},
            )
        pending = len(pending_approvals)
        suffix = (
            "\nExisting pending approvals still require a decision."
            if pending
            else ""
        )
        await send(chat_id, f"Approval: {mode}" + suffix)
        return True

    if cmd == "/todos":
        await send(
            chat_id,
            format_todos(),
        )

        return True

    if cmd == "/subagents":
        await send(chat_id, format_subagents())
        return True

    if cmd == "/models":
        provider_id = arg.strip() or session.current_model_target().provider_id
        try:
            provider = MODEL_CONFIGURATION.provider(provider_id)
            models = await list_models(provider_id)
        except Exception as exc:
            await send(
                chat_id,
                f"Models error for {provider_id}: {type(exc).__name__}: {exc}",
            )
            return True
        current = session.current_model_target()
        lines = [
            f"{'*' if provider_id == current.provider_id and item == current.model else ' '} {item}"
            for item in models
        ]
        header = f"Models from {provider.id}:"
        await send(chat_id, header + ("\n" + "\n".join(lines) if lines else " none"))
        return True

    if cmd == "/model":
        if not arg:
            current = session.current_model_target()
            providers = "\n".join(
                f"  {provider_id} (default: {provider.default_model or 'none'})"
                for provider_id, provider in sorted(
                    MODEL_CONFIGURATION.providers.items()
                )
            )
            await send(
                chat_id,
                (
                    f"Model target: {current.provider_id} / {current.model}\n"
                    f"Providers:\n{providers}\n"
                    "Usage: /model <provider> <model>\n"
                    "Use /models [provider] to list served models."
                ),
            )
            return True
        if thread_active():
            await send(chat_id, "Clear or merge the active thread first.")
            return True
        if session_transition_busy():
            await send(
                chat_id,
                "Busy. Use /stop all first.",
            )
            return True
        parts = arg.split(maxsplit=1)
        if len(parts) != 2:
            await send(
                chat_id,
                "Usage: /model <provider> <model>",
            )
            return True
        provider_id, model = parts
        try:
            selected = MODEL_CONFIGURATION.select_target(
                parent=session.current_model_target(),
                provider_id=provider_id,
                model=model,
            )
            model_provider(selected).api_key()
        except (RuntimeError, ValueError) as exc:
            await send(chat_id, f"Model target error: {exc}")
            return True
        old_session_id = str(state.get("current_session_id") or "")
        if old_session_id and session_exists(old_session_id):
            end_session(
                old_session_id,
                reason="model_target_changed",
                todos=list(state.get("todos") or []),
            )
        new_session_id = create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=selected.to_dict(),
        )
        session.set_current_model_target(selected)
        state["current_session_id"] = new_session_id
        state["todos"] = []
        state["pending_inputs"] = []
        state["active_root_turn"] = None
        state["interrupted_subagents"] = []
        save_state()
        await send(
            chat_id,
            (
                f"Model target: {selected.provider_id} / {selected.model}\n"
                "Started a fresh session; the previous transcript was archived."
            ),
        )
        return True

    return False
